//===- CFGPrinter.cpp - DOT printer for the control flow graph ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a `-dot-cfg` analysis pass, which emits the
// `<prefix>.<fnname>.dot` file for each function in the program, with a graph
// of the CFG for that function. The default value for `<prefix>` is `cfg` but
// can be customized as needed.
//
// The other main feature of this file is that it implements the
// Function::viewCFG method, which is useful for debugging passes which operate
// on the CFG.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
using namespace llvm;

static cl::opt<std::string> CFGFuncName(
    "cfg-func-name", cl::Hidden,
    cl::desc("The name of a function (or its substring)"
             " whose CFG is viewed/printed."));

static cl::opt<std::string> CFGDotFilenamePrefix(
    "cfg-dot-filename-prefix", cl::Hidden,
    cl::desc("The prefix used for the CFG dot file names."));

namespace {
  struct CFGViewerLegacyPass : public FunctionPass {
    static char ID; // Pass identifcation, replacement for typeid
    CFGViewerLegacyPass() : FunctionPass(ID) {
      initializeCFGViewerLegacyPassPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override {
      F.viewCFG();
      return false;
    }

    void print(raw_ostream &OS, const Module* = nullptr) const override {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }
  };
}

char CFGViewerLegacyPass::ID = 0;
INITIALIZE_PASS(CFGViewerLegacyPass, "view-cfg", "View CFG of function", false, true)

PreservedAnalyses CFGViewerPass::run(Function &F,
                                     FunctionAnalysisManager &AM) {
  F.viewCFG();
  return PreservedAnalyses::all();
}


namespace {
  struct CFGOnlyViewerLegacyPass : public FunctionPass {
    static char ID; // Pass identifcation, replacement for typeid
    CFGOnlyViewerLegacyPass() : FunctionPass(ID) {
      initializeCFGOnlyViewerLegacyPassPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override {
      F.viewCFGOnly();
      return false;
    }

    void print(raw_ostream &OS, const Module* = nullptr) const override {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }
  };
}

char CFGOnlyViewerLegacyPass::ID = 0;
INITIALIZE_PASS(CFGOnlyViewerLegacyPass, "view-cfg-only",
                "View CFG of function (with no function bodies)", false, true)

PreservedAnalyses CFGOnlyViewerPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  F.viewCFGOnly();
  return PreservedAnalyses::all();
}

struct DJWriter : public GraphWriter<const Function *> {
  DominatorTree &DT;
  DJWriter(raw_ostream &O, const Function *F, bool SN, DominatorTree &DT)
      : GraphWriter<const Function *>(O, F, SN), DT(DT) {}

  void writeEdge(NodeRef Node, unsigned edgeidx, child_iterator EI) override {
    if (NodeRef TargetNode = *EI) {
      int DestPort = -1;
      if (DTraits.edgeTargetsEdgeSource(Node, EI)) {
        child_iterator TargetIt = DTraits.getEdgeTarget(Node, EI);

        // Figure out which edge this targets...
        unsigned Offset =
            (unsigned)std::distance(GTraits::child_begin(TargetNode), TargetIt);
        DestPort = static_cast<int>(Offset);
      }

      if (DTraits.getEdgeSourceLabel(Node, EI).empty())
        edgeidx = -1;

      std::string Attr = DTraits.getEdgeAttributes(Node, EI, G);
      Attr += "; color=green";

      emitEdge(static_cast<const void *>(Node), edgeidx,
               static_cast<const void *>(TargetNode), DestPort, Attr);
    }
  }
};

static void writeCFGToDotFile(Function &F, bool CFGOnly = false,
                              DominatorTree *DT = nullptr) {
  if (!CFGFuncName.empty() && !F.getName().contains(CFGFuncName))
    return;
  std::string Filename =
      (CFGDotFilenamePrefix + "." + F.getName() + ".dot").str();
  errs() << "Writing '" << Filename << "'...";

  std::error_code EC;
  raw_fd_ostream File(Filename, EC, sys::fs::F_Text);

  if (!EC) {
    // Start the graph emission process...
    if (DT) {
      DJWriter W(File, &F, CFGOnly, *DT);
      // Emit the graph.
      W.writeGraph("");
    } else {
      GraphWriter<const Function *> W(File, &F, CFGOnly);
      // Emit the graph.
      W.writeGraph("");
    }
  } else
    errs() << "  error opening file for writing!";
  errs() << "\n";
}

namespace {
  struct CFGPrinterLegacyPass : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    CFGPrinterLegacyPass() : FunctionPass(ID) {
      initializeCFGPrinterLegacyPassPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override {
      writeCFGToDotFile(F);
      return false;
    }

    void print(raw_ostream &OS, const Module* = nullptr) const override {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }
  };
}

char CFGPrinterLegacyPass::ID = 0;
INITIALIZE_PASS(CFGPrinterLegacyPass, "dot-cfg", "Print CFG of function to 'dot' file",
                false, true)

PreservedAnalyses CFGPrinterPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  writeCFGToDotFile(F, /* CFGOnly */ false, &DT);
  return PreservedAnalyses::all();
}

namespace {
  struct CFGOnlyPrinterLegacyPass : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    CFGOnlyPrinterLegacyPass() : FunctionPass(ID) {
      initializeCFGOnlyPrinterLegacyPassPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override {
      writeCFGToDotFile(F, /*CFGOnly=*/true);
      return false;
    }
    void print(raw_ostream &OS, const Module* = nullptr) const override {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }
  };
}

char CFGOnlyPrinterLegacyPass::ID = 0;
INITIALIZE_PASS(CFGOnlyPrinterLegacyPass, "dot-cfg-only",
   "Print CFG of function to 'dot' file (with no function bodies)",
   false, true)

PreservedAnalyses CFGOnlyPrinterPass::run(Function &F,
                                          FunctionAnalysisManager &AM) {
  writeCFGToDotFile(F, /*CFGOnly=*/true);
  return PreservedAnalyses::all();
}

/// viewCFG - This function is meant for use from the debugger.  You can just
/// say 'call F->viewCFG()' and a ghostview window should pop up from the
/// program, displaying the CFG of the current function.  This depends on there
/// being a 'dot' and 'gv' program in your path.
///
void Function::viewCFG() const {
  if (!CFGFuncName.empty() && !getName().contains(CFGFuncName))
     return;
  ViewGraph(this, "cfg" + getName());
}

/// viewCFGOnly - This function is meant for use from the debugger.  It works
/// just like viewCFG, but it does not include the contents of basic blocks
/// into the nodes, just the label.  If you are only interested in the CFG
/// this can make the graph smaller.
///
void Function::viewCFGOnly() const {
  if (!CFGFuncName.empty() && !getName().contains(CFGFuncName))
     return;
  ViewGraph(this, "cfg" + getName(), true);
}

FunctionPass *llvm::createCFGPrinterLegacyPassPass () {
  return new CFGPrinterLegacyPass();
}

FunctionPass *llvm::createCFGOnlyPrinterLegacyPassPass () {
  return new CFGOnlyPrinterLegacyPass();
}

