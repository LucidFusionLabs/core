Name "quake"
OutFile "quake-installer.exe"
InstallDir $PROGRAMFILES\com.lucidfusionlabs\quake

Page directory
Page instfiles

Section "";

RmDir /r "$INSTDIR"

SetOutPath "$INSTDIR"
File "Debug\quake.exe"
File "Debug\*.dll"

SetOutPath "$INSTDIR\assets"
File "assets\*"

SetOutPath "$INSTDIR"
CreateDirectory "$SMPROGRAMS\com.lucidfusionlabs"
createShortCut "$SMPROGRAMS\com.lucidfusionlabs\quake.lnk" "$INSTDIR\quake.exe"

SectionEnd
 
