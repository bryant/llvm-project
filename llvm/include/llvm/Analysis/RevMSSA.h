#ifndef LLVM_ANALYSIS_REVMSSA_H
#define LLVM_ANALYSIS_REVMSSA_H

#include "llvm/Analysis/MemorySSA.h"

namespace rmssa {

struct Access {};

struct Def : public Access {
  Value *V;
};

struct Lambda : public Def {
  BasicBlock &getBlock() const { return *cast<BasicBlock>(V); }
};

struct Use : public Def {
  Def *Def;
};

struct RevForm {
  Def LiveOnExit;
};

} // namespace rmssa

#endif
