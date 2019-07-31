add_llvm_library(LLVMXCoreDesc
  XCoreInstPrinter.cpp
  XCoreMCTargetDesc.cpp
  XCoreMCAsmInfo.cpp

  LINK_COMPONENTS

  MC
  Support
  XCoreInfo
  )
