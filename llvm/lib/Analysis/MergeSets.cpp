// NVIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA_COPYRIGHT_END

// Computes the merge sets (really just the IDF) of each basic block, but with
// average complexity that is linear to number of blocks.

#define DEBUG_TYPE "merge-sets"
#define PASS_NAME "Compute merge sets"

#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

struct MergeSets {
  const DominatorTree *DT = nullptr;
  // The core reason for this analysis' existence. Record of each BB's merge
  // sets and canonical bit position (currently BFS order). Deliberately not
  // storing DomTreeNode as that could require an extra look-up from clients.
  DenseMap<BasicBlock *, std::pair<unsigned, BitVector>> Inner;
  SmallVector<const DomTreeNode *, 32> BfsVec;

  const DominatorTree &getDomTree() const { return *DT; }

  BitVector getMergeBitVec(const BasicBlock &BB) const;

  unsigned getIndex(const BasicBlock &BB) const;

  bool tdmsc1(ArrayRef<const DomTreeNode *> BfsVec) {
    DenseSet<std::pair<const DomTreeNode *, const DomTreeNode *>> Visited;
    bool Inconsistent = false;

    auto propagateToShadow = [&](const DomTreeNode &P, const DomTreeNode &S) {
      const DomTreeNode *L = &P;
      do {
        assert(Inner.count(L->getBlock()) && Inner.count(S.getBlock()) &&
               "Should have initialized Inner with BfsVec.");
        auto Lt = Inner.find(L->getBlock());
        auto St = Inner.find(S.getBlock());
        l.merge |= s.merge | s;
        L = L->getIDom();
      } while (L->getIDom() != S.getIDom());
    };

    for (unsigned Idx = 0; Idx < BfsVec.size(); Idx += 1) {
      for (const DomTreeNode &JPred : jpreds(BfsVec[Idx]->getBlock())) {
        auto Pair = Visited.insert({&JPred, &BfsVec[Idx]});
        if (!Pair.second)
          continue;
        const DomTreeNode &L = *propagateToShadow(JPred, BfsVec[Idx]);
        Inconsistent = checkInconsistency(L);
      }
    }

    return Inconsistent;
  }

  MergeSets &recompute(const DominatorTree &DT_) {
    Inner.clear();
    DT = &DT_;

    computeBFS();

    // TODO: Impl and bench TDMSC 2.
    for (bool AnotherOne = true; AnotherOne; AnotherOne = tdmsc1())
      ;
  }
};

struct MergeSetsPass : public FunctionPass {
  static char ID;
  MergeSets MS;

  MergeSetsPass() : FunctionPass(ID) {
    initializeMergeSetsPassPass(*PassRegistry::getPassRegistry());
  }

  MergeSets &getMS() { return MS; }

  bool runOnFunction(Function &) override {
    MS.recompute(getAnalysis<DominatorTreeWrapperPass>().getDomTree());
    return false;
  }
};

char MergeSetsPass::ID = 0;

INITIALIZE_PASS_BEGIN(MergeSetsPass, DEBUG_TYPE, PASS_NAME, true, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(MergeSetsPass, DEBUG_TYPE, PASS_NAME, true, true)

Pass *llvm::createMergeSetsPass() { return new MergeSetsPass(); }
