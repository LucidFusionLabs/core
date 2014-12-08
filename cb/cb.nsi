Name "cb"
OutFile "cb-installer.exe"
InstallDir $PROGRAMFILES\com.lucidfusionlabs\cb

Page directory
Page instfiles

Section "";

RmDir /r "$INSTDIR"

SetOutPath "$INSTDIR"
File "Debug\cb.exe"
File "Debug\*.dll"

SetOutPath "$INSTDIR\assets"
File "assets\*"

SetOutPath "$INSTDIR"
CreateDirectory "$SMPROGRAMS\com.lucidfusionlabs"
createShortCut "$SMPROGRAMS\com.lucidfusionlabs\cb.lnk" "$INSTDIR\cb.exe"

SectionEnd
 
