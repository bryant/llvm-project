set(LLVM_LINK_COMPONENTS
  AsmParser
  BinaryFormat
  CodeGen
  Core
  MC
  Support
  Target
  )

add_llvm_library(LLVMMIRParser
  MILexer.cpp
  MIParser.cpp
  MIRParser.cpp

  DEPENDS
  intrinsics_gen
  )
