Name "master"
OutFile "master-installer.exe"
InstallDir $PROGRAMFILES\com.lucidfusionlabs\master

Page directory
Page instfiles

Section "";

RmDir /r "$INSTDIR"

SetOutPath "$INSTDIR"
File "Debug\master.exe"
File "Debug\*.dll"

SetOutPath "$INSTDIR\assets"
File "assets\*"

SetOutPath "$INSTDIR"
CreateDirectory "$SMPROGRAMS\com.lucidfusionlabs"
createShortCut "$SMPROGRAMS\com.lucidfusionlabs\master.lnk" "$INSTDIR\master.exe"

SectionEnd
 
