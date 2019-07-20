set(LLVM_LINK_COMPONENTS
  MC
  MCParser
  RISCVDesc
  RISCVInfo
  RISCVUtils
  Support
  )

add_llvm_library(LLVMRISCVAsmParser
  RISCVAsmParser.cpp
  )
