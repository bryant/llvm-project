add_llvm_library(LLVMARMAsmParser
  ARMAsmParser.cpp

  LINK_COMPONENTS

  ARMDesc
  ARMInfo
  ARMUtils
  MC
  MCParser
  Support
  )
