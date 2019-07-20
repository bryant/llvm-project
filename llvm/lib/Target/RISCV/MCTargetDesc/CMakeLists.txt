set(LLVM_LINK_COMPONENTS
  MC
  RISCVInfo
  RISCVUtils
  Support
  )

add_llvm_library(LLVMRISCVDesc
  RISCVAsmBackend.cpp
  RISCVELFObjectWriter.cpp
  RISCVInstPrinter.cpp
  RISCVMCAsmInfo.cpp
  RISCVMCCodeEmitter.cpp
  RISCVMCExpr.cpp
  RISCVMCTargetDesc.cpp
  RISCVTargetStreamer.cpp
  RISCVELFStreamer.cpp
)
