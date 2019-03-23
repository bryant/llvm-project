#ifndef LLVM_TRANSFORMS_SCALAR_DSE_H
#define LLVM_TRANSFORMS_SCALAR_DSE_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class Function;

struct DSEPass : public PassInfoMixin<DSEPass> {
  PreservedAnalyses run(Function &F, AnalysisManager<Function> &AM);
};
} // namespace llvm

#endif
