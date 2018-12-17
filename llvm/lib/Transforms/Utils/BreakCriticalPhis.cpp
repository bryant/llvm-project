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

// \file Splits critical phi edges only where needed. This is primarily needed
// in D2IR to ensure that vreg copies (from phi lowering) are post-dominated by
// the lowered phi block (otherwise, there would be a path from the copy inst to
// exit that doesn't cross the phi, IE unnecessary copy).
//
// If there's no phi, the critical edge won't matter and we don't split to avoid
// unnecessary jumps in ORI.
//
// If the critical edge happens to be a back edge, we really don't want to split
// because this de-canonicalizes the containing loop (if it were already
// rotated) so that re-canonicalization is impossible and thus inhibits further
// OCG loop opts.
//
// TODO: When there are multiple identical edges, it's enough to create just one
// split node for all edges rather than one for each (which is what we do now).
// See below occurrences of MergeIdenticalEdges.
//
// TODO: Hook this up to the new pass manager and/or expose to opt in order to
// unit-test properly.

#define DEBUG_TYPE "break-crit-phis"

#include "nvvm/Transforms/BreakCriticalPhis.h"

#include "nvvm/InitializePasses.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#if NVVM_VERSION < 600
#include "llvm/Analysis/Dominators.h"
#else
#include "llvm/IR/Dominators.h"
#endif

using namespace llvm;

// Shorthand to print an edge to raw_ostream.
struct edge {
  const BasicBlock *From, *To;
  edge(const BasicBlock &F, const BasicBlock &T) : From(&F), To(&T) {}
  friend raw_ostream &operator<<(raw_ostream &O, const edge &E) {
    return O << E.From->getName() << ", " << E.To->getName();
  }
};

static bool isEHPad(const BasicBlock &BB) {
#if NVVM_VERSION < 600
  return BB.isLandingPad();
#else
  return BB.isEHPad();
#endif
}

#if NVVM_VERSION < 600
// Provide a const-correct version of GetSuccessorNumber (fixed in r253733). In
// all versions, neither BB is modified.
static unsigned GetSuccessorNumber(const BasicBlock *A, const BasicBlock *B) {
  return llvm::GetSuccessorNumber(const_cast<BasicBlock *>(A),
                                  const_cast<BasicBlock *>(B));
}
#endif

bool isBackEdge(const BasicBlock &From, const BasicBlock &To,
                const DominatorTree &DT) {
  return &To == &From || DT.dominates(&To, &From);
}

// Return true iff BB has at least one phi, multiple preds, and at least one
// multiple-succ pred. Used to verify results of BreakCriticalPhis.
bool hasCriticalPhis(const BasicBlock &BB, const DominatorTree &DT) {
  if (BB.empty() || !isa<PHINode>(BB.front()) || isEHPad(BB))
    return false;

  // TODO: (NVVM60) llvm::predecessors() shorthand.
  for (const BasicBlock *Pred : make_range(pred_begin(&BB), pred_end(&BB))) {
    if (isBackEdge(*Pred, BB, DT))
      // Was a back edge. Of course we did not split.
      continue;

    if (!isa<IndirectBrInst>(Pred->getTerminator()) &&
        isCriticalEdge(Pred->getTerminator(), GetSuccessorNumber(Pred, &BB),
                       /* MergeIdenticalEdges = */ false)) {
      LLVM_DEBUG(dbgs() << "Critical phi (" << edge(*Pred, BB) << ")\n");
      return true;
    }
  }

  return false;
}

struct BreakCriticalPhis : public FunctionPass {
  static char ID;

  BreakCriticalPhis() : FunctionPass(ID) {
    initializeBreakCriticalPhisPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
#if NVVM_VERSION < 600
    using DominatorTreeWrapperPass = DominatorTree;
#endif
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }

  BasicBlock *splitCritEdge(TerminatorInst &TI, unsigned SuccIdx,
                            DominatorTree &DT, LoopInfo *LI) {
#if NVVM_VERSION < 600
    return SplitCriticalEdge(&TI, SuccIdx, this,
                             /* MergeIdenticalEdges = */ false);
#else
    CriticalEdgeSplittingOptions Opts(&DT, LI);
    return SplitCriticalEdge(&TI, SuccIdx, Opts.clearPreserveLoopSimplify());
#endif
  }
};

char BreakCriticalPhis::ID = 0;

INITIALIZE_PASS_BEGIN(BreakCriticalPhis, "break-crit-phis",
                      "D2IR Break Critical Phis", false, false)
#if NVVM_VERSION < 600
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
#else
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
#endif
INITIALIZE_PASS_END(BreakCriticalPhis, "break-crit-phis",
                    "D2IR Break Critical Phis", false, false)

FunctionPass *llvm::createBreakCriticalPhisPass() {
  return new BreakCriticalPhis();
}

bool BreakCriticalPhis::runOnFunction(Function &F) {
#if NVVM_VERSION < 600
  LoopInfo *LI = getAnalysisIfAvailable<LoopInfo>();
  DominatorTree &DT = getAnalysis<DominatorTree>();
#else
  if (skipFunction(F))
    return false;

  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  // Preserve LoopInfo if it's been built already.
  LoopInfo *LI = nullptr;
  if (auto *LIWP = getAnalysisIfAvailable<LoopInfoWrapperPass>())
    LI = &LIWP->getLoopInfo();
#endif

  bool Changed = false;

  // Look at phi-containing succs of multiple-succ blocks. If there are no phis
  // in the critical edge succ, then we never need to split. If succ has a phi
  // but only one pred, SplitCriticalEdge will catch this and refuse to split,
  // which is also okay.
  for (BasicBlock &BB : F) {
    TerminatorInst &TI = *BB.getTerminator();
    if (TI.getNumSuccessors() <= 1 || isa<IndirectBrInst>(&TI))
      continue;

    for (unsigned SuccIdx = 0; SuccIdx < TI.getNumSuccessors(); SuccIdx += 1) {
      BasicBlock &Succ = *TI.getSuccessor(SuccIdx);

      // Should we split?
      if (!Succ.empty() && isa<PHINode>(Succ.front()) &&
          !isBackEdge(BB, Succ, DT)) {
        LLVM_DEBUG(dbgs() << "Attempting to split (" << edge(BB, Succ)
                          << ")\n");
        // Yes, try to split. SplitCriticalEdge itself could still refuse to.
        if (BasicBlock *NewNode = splitCritEdge(TI, SuccIdx, DT, LI)) {
          LLVM_DEBUG(dbgs() << "Split node is " << NewNode->getName() << "\n");
          Changed = true;
        }
      } else
        LLVM_DEBUG({
          if (isBackEdge(BB, Succ, DT))
            dbgs() << "Skipping back edge (" << edge(BB, Succ) << ")\n";
        });
    }
  }

  // TODO: (NVVM60) llvm::all_of shorthand.
  assert(std::all_of(
             F.begin(), F.end(),
             [&](const BasicBlock &BB) { return !hasCriticalPhis(BB, DT); }) &&
         "Discovered critical phi edge.");

  return Changed;
}
