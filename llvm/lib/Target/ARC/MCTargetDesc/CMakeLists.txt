set(LLVM_LINK_COMPONENTS
  ARCInfo
  MC
  Support
  )

add_llvm_library(LLVMARCDesc
  ARCInstPrinter.cpp
  ARCMCTargetDesc.cpp
  ARCMCAsmInfo.cpp
  )
