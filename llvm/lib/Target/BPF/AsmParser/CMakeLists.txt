add_llvm_library(LLVMBPFAsmParser
  BPFAsmParser.cpp

  LINK_COMPONENTS

  BPFDesc
  BPFInfo
  MC
  MCParser
  Support
  )
