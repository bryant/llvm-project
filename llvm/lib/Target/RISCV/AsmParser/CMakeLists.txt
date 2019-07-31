add_llvm_library(LLVMRISCVAsmParser
  RISCVAsmParser.cpp

  LINK_COMPONENTS

  MC
  MCParser
  RISCVDesc
  RISCVInfo
  RISCVUtils
  Support
  )
