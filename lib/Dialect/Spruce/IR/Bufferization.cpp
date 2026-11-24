#include "spruce/Dialect/Spruce/IR/SpruceDialect.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::spruce;

namespace {

// The conversion emits an isolated kernel wrapping one MMA. Its result is
// equivalent to input 2 only when that MMA updates the matching block argument.
MmaOp getTileMma(KernelOp kernel) {
  if (!kernel || kernel.getNumOperands() != 3 || kernel.getNumResults() != 1)
    return {};
  Block &body = kernel.getBody().front();
  if (body.getOperations().size() != 2)
    return {};
  auto mma = dyn_cast<MmaOp>(body.front());
  if (!mma || !mma.getResult() || mma.getOperands() != body.getArguments() ||
      cast<YieldOp>(body.back()).getOperand(0) != mma.getResult() ||
      mma.getResult().getType() != mma.getAcc().getType())
    return {};
  return mma;
}

FailureOr<SmallVector<Value>>
getOperandBuffers(Operation *op, RewriterBase &rewriter,
                  const BufferizationOptions &options,
                  BufferizationState &state) {
  SmallVector<Value> buffers;
  for (Value operand : op->getOperands()) {
    auto buffer = getBuffer(rewriter, operand, options, state);
    if (failed(buffer))
      return failure();
    buffers.push_back(*buffer);
  }
  return buffers;
}

void buildBufferMma(RewriterBase &rewriter, MmaOp mma, ValueRange buffers) {
  auto newMma = MmaOp::create(rewriter, mma.getLoc(), Type{}, buffers[0],
                              buffers[1], buffers[2]);
  newMma->setAttrs(mma->getAttrs());
}

struct MmaModel : BufferizableOpInterface::ExternalModel<MmaModel, MmaOp> {
  LogicalResult verifyAnalysis(Operation *op, const AnalysisState &) const {
    for (Type type : op->getOperandTypes()) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      if (tensor && tensor.getEncoding())
        return op->emitOpError("bufferization requires unencoded tensor tiles");
    }
    return success();
  }

  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &operand,
                               const AnalysisState &) const {
    // The kernel models this write when it bufferizes the whole region.
    return operand.getOperandNumber() == 2 &&
           !getTileMma(dyn_cast<KernelOp>(op->getParentOp()));
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &operand,
                                      const AnalysisState &) const {
    if (operand.getOperandNumber() == 2)
      return {{op->getResult(0), BufferRelation::Equivalent}};
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    if (getTileMma(dyn_cast<KernelOp>(op->getParentOp())))
      return success();
    auto mma = cast<MmaOp>(op);
    if (mma.getResult().getType() != mma.getAcc().getType())
      return mma.emitOpError(
          "bufferization requires matching accumulator and result types");
    auto buffers = getOperandBuffers(op, rewriter, options, state);
    if (failed(buffers))
      return failure();
    buildBufferMma(rewriter, mma, *buffers);
    replaceOpWithBufferizedValues(rewriter, op, (*buffers)[2]);
    return success();
  }
};

struct KernelModel
    : BufferizableOpInterface::ExternalModel<KernelModel, KernelOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &operand,
                               const AnalysisState &) const {
    return operand.getOperandNumber() == 2 && getTileMma(cast<KernelOp>(op));
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &operand,
                                      const AnalysisState &) const {
    if (operand.getOperandNumber() == 2 && getTileMma(cast<KernelOp>(op)))
      return {{op->getResult(0), BufferRelation::Equivalent}};
    return {};
  }

  LogicalResult verifyAnalysis(Operation *op, const AnalysisState &) const {
    if (bufferization::hasTensorSemantics(op) &&
        !getTileMma(cast<KernelOp>(op)))
      return op->emitOpError(
          "bufferization requires a kernel yielding a single tile MMA");
    return success();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto kernel = cast<KernelOp>(op);
    auto mma = getTileMma(kernel);
    if (!mma)
      return failure();
    auto buffers = getOperandBuffers(op, rewriter, options, state);
    if (failed(buffers))
      return failure();
    auto loc = op->getLoc();
    auto bufferKernel = KernelOp::create(
        rewriter, loc, TypeRange{(*buffers)[2].getType()}, *buffers);
    bufferKernel->setAttrs(op->getAttrs());
    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto *body = rewriter.createBlock(
          &bufferKernel.getBody(), {}, bufferKernel->getOperandTypes(),
          SmallVector<Location>(buffers->size(), loc));
      buildBufferMma(rewriter, mma, body->getArguments());
      YieldOp::create(rewriter, kernel.getBody().front().back().getLoc(),
                      body->getArgument(2));
    }
    replaceOpWithBufferizedValues(rewriter, op, bufferKernel.getResults());
    return success();
  }
};

struct YieldModel
    : BufferizableOpInterface::ExternalModel<YieldModel, YieldOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *, RewriterBase &,
                          const BufferizationOptions &,
                          BufferizationState &) const {
    // The enclosing kernel consumes the yielded tile and replaces the
    // terminator.
    return success();
  }
};
} // namespace

void SpruceDialect::registerBufferizationInterfaces() {
  KernelOp::attachInterface<KernelModel>(*getContext());
  YieldOp::attachInterface<YieldModel>(*getContext());
  MmaOp::attachInterface<MmaModel>(*getContext());
}
