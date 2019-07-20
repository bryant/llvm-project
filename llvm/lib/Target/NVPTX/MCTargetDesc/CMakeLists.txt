set(LLVM_LINK_COMPONENTS
  MC
  NVPTXInfo
  Support
  )

add_llvm_library(LLVMNVPTXDesc
  NVPTXInstPrinter.cpp
  NVPTXMCAsmInfo.cpp
  NVPTXMCTargetDesc.cpp
  NVPTXTargetStreamer.cpp
  )
