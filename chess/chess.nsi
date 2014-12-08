Name "chess"
OutFile "chess-installer.exe"
InstallDir $PROGRAMFILES\com.lucidfusionlabs\chess

Page directory
Page instfiles

Section "";

RmDir /r "$INSTDIR"

SetOutPath "$INSTDIR"
File "Debug\chess.exe"
File "Debug\*.dll"

SetOutPath "$INSTDIR\assets"
File "assets\*"

SetOutPath "$INSTDIR"
CreateDirectory "$SMPROGRAMS\com.lucidfusionlabs"
createShortCut "$SMPROGRAMS\com.lucidfusionlabs\chess.lnk" "$INSTDIR\chess.exe"

SectionEnd
 
