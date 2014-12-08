Name "image"
OutFile "image-installer.exe"
InstallDir $PROGRAMFILES\com.lucidfusionlabs\image

Page directory
Page instfiles

Section "";

RmDir /r "$INSTDIR"

SetOutPath "$INSTDIR"
File "Debug\image.exe"
File "Debug\*.dll"

SetOutPath "$INSTDIR\assets"
File "assets\*"

SetOutPath "$INSTDIR"
CreateDirectory "$SMPROGRAMS\com.lucidfusionlabs"
createShortCut "$SMPROGRAMS\com.lucidfusionlabs\image.lnk" "$INSTDIR\image.exe"

SectionEnd
 
