Name "market"
OutFile "market-installer.exe"
InstallDir $PROGRAMFILES\com.lucidfusionlabs\market

Page directory
Page instfiles

Section "";

RmDir /r "$INSTDIR"

SetOutPath "$INSTDIR"
File "Debug\market.exe"
File "Debug\*.dll"

SetOutPath "$INSTDIR\assets"
File "assets\*"

SetOutPath "$INSTDIR"
CreateDirectory "$SMPROGRAMS\com.lucidfusionlabs"
createShortCut "$SMPROGRAMS\com.lucidfusionlabs\market.lnk" "$INSTDIR\market.exe"

SectionEnd
 
