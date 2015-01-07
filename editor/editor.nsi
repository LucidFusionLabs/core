Name "editor"
OutFile "editor-installer.exe"
InstallDir $PROGRAMFILES\com.lucidfusionlabs\editor

Page directory
Page instfiles

Section "";

RmDir /r "$INSTDIR"

SetOutPath "$INSTDIR"
File "Debug\editor.exe"
File "Debug\*.dll"

SetOutPath "$INSTDIR\assets"
File "assets\*"

SetOutPath "$INSTDIR"
CreateDirectory "$SMPROGRAMS\com.lucidfusionlabs"
createShortCut "$SMPROGRAMS\com.lucidfusionlabs\editor.lnk" "$INSTDIR\editor.exe"

SectionEnd
 
