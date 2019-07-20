set(LLVM_LINK_COMPONENTS
  BPFDesc
  BPFInfo
  MC
  MCParser
  Support
  )

add_llvm_library(LLVMBPFAsmParser
  BPFAsmParser.cpp
  )
