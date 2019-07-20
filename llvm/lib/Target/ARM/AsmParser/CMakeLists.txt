set(LLVM_LINK_COMPONENTS
  ARMDesc
  ARMInfo
  ARMUtils
  MC
  MCParser
  Support
  )

add_llvm_library(LLVMARMAsmParser
  ARMAsmParser.cpp
  )
