#ifndef LLVM_TRANSFORMS_SCALAR_SPARSEDSE_H
#define LLVM_TRANSFORMS_SCALAR_SPARSEDSE_H

#include "llvm/IR/PassManager.h"

namespace llvm {
struct SparseDSEPass : public PassInfoMixin<SparseDSEPass> {
  PreservedAnalyses run(Function &, AnalysisManager<Function> &);
};
} // namespace llvm

#endif
