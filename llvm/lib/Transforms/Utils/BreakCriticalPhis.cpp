// NVIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
// NVIDIA_COPYRIGHT_END

// \file Splits critical phi edges but avoids loop back edges. TODO: Hook this
// up to the new pass manager and expose to opt in order to unit-test this
// properly.

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/BreakCriticalPhis.h"

#define DEBUG_TYPE "break-crit-phis"

using namespace llvm;

struct BreakCriticalPhis : public FunctionPass {
  static char ID;

  BreakCriticalPhis() : FunctionPass(ID) {
    initializeBreakCriticalPhisPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
  }
};

char BreakCriticalPhis::ID = 0;

INITIALIZE_PASS_BEGIN(BreakCriticalPhis, "break-crit-phis",
                      "D2IR Break Critical Edges", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(BreakCriticalPhis, "break-crit-phis",
                    "D2IR Break Critical Edges", false, false)

FunctionPass *llvm::createBreakCriticalPhisPass() {
  return new BreakCriticalPhis();
}

bool BreakCriticalPhis::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  using Edge = std::pair<BasicBlock *, BasicBlock *>;
  // Don't split the same block pair more than once, esp. in cases where
  // multiple edges exist between the same two blocks.
  DenseSet<Edge> ToSplit;
  bool Changed = false;

  for (BasicBlock &BB : F) {
    if (BB.empty() || !isa<PHINode>(BB.front()))
      continue;

    Loop *BBL = LI.getLoopFor(&BB);

    for (BasicBlock *Pred : make_range(pred_begin(&BB), pred_end(&BB))) {
      // We don't want to split critical back edges because it de-canonicalizes
      // loops that OCG would otherwise need to unroll and in general is just
      // bad for code placement.
      auto IsBackEdge = [&]() {
        return BBL && BBL->getHeader() == &BB && LI.getLoopFor(Pred) == BBL;
      };

      // Assumption: Useless phis have been removed and presence of phis in BB
      // implies having multiple preds. So this is a critical edge iff pred
      // block has multiple succs.
      if (Pred->getTerminator()->getNumSuccessors() > 1 && !IsBackEdge()) {
        LLVM_DEBUG(dbgs() << "Queueing edge (" << Pred->getName() << ", "
                          << BB.getName() << ")\n");
        ToSplit.insert({Pred, &BB});
      } else if (IsBackEdge())
        LLVM_DEBUG(dbgs() << "Skipping back edge (" << Pred->getName() << ", "
                          << BB.getName() << ")\n");
    }
  }

  for (const Edge &E : ToSplit) {
    // Note that SplitCriticalEdge could still choose not to split this (e.g. if
    // the succ block were a landing pad), in which case a nullptr would be
    // returned.
    CriticalEdgeSplittingOptions Opts(/* DominatorTree * = */ nullptr,
                                      /* LoopInfo * = */ &LI);
    Opts.setMergeIdenticalEdges();
    if (BasicBlock *NewNode = SplitCriticalEdge(E.first, E.second, Opts)) {
      LLVM_DEBUG(dbgs() << "Splitting edge (" << E.first->getName() << ", "
                        << E.second->getName() << "); split node is "
                        << NewNode->getName() << "\n");
      Changed = true;
    }
  }

  return Changed;
}
