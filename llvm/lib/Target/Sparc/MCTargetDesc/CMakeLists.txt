add_llvm_library(LLVMSparcDesc
  SparcAsmBackend.cpp
  SparcELFObjectWriter.cpp
  SparcInstPrinter.cpp
  SparcMCAsmInfo.cpp
  SparcMCCodeEmitter.cpp
  SparcMCTargetDesc.cpp
  SparcMCExpr.cpp
  SparcTargetStreamer.cpp

  LINK_COMPONENTS

  MC
  SparcInfo
  Support
  )
