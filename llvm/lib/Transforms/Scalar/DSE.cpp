// Think about this from load elim point of view.

#include "llvm/Transforms/Scalar/DSE.h"

bool elimDeadStores(rmssa::Def &D) {
  SmallVector<Use *, 16> Uses{D.begin(), D.end()};
  // Sort by PDO-ordered RPO.
  sort(Uses, [](const Use &U) {});
}

bool elimDeadStores(const RevForm &RF) {
  return std::accumulate(
      RF.defs_begin(), RF.defs_end(), false,
      [](bool Changed, rmssa::Def &D) { return elimDeadStores(D) | Changed; });
}

PreservedAnalyses DSEPass::run(Function &F, AnalysisManager<Function> &AM) {}
