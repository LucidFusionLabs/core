http://lucidfusionlabs.com/svn/lfl/new_app_template/README.txt
==============================================================

OVERVIEW
--------

Run: lfl/new_app_template/clone.sh <organization name> <package name> <binary name>

Eg: ./new_app_template/clone.sh com.lucidfusionlabs SpaceballFuture spaceball
    Creates ./spaceball/spaceball.cpp, ./spaceball/spaceball-android, ./spaceball/spaceball-iphone, etc

Append "add_subdiretory(spaceball)" to lfl/CMakeLists.txt.


MANIFEST
--------

- CMakeLists.txt:                         Build rules
- new_app_template.cpp:                   App source code
- assets/:                                All models, multimedia, databases, etc here  

- assets/icon.ico:                        Windows icon file
- assets/icon.icns:                       Mac icon file
- assets/icon.bmp:                        32x32 Bitmap icon file

        Icon converter: http://iconverticons.com/

- *-android/res/drawable-hdpi/icon.png:   72x72 Android icon
- *-android/res/drawable-mdpi/icon.png:   48x48 Android icon
- *-android/res/drawable-ldpi/icon.png:   36x36 Android icon
- *-iphone/Images/Icon.png:               57x57 iPhone icon

        Png resizer http://images.my-addr.com/resize_png_online_tool-free_png_resizer_for_web.php

- *-iphone/Images/Default.png:            320x480 iPhone splash screen
- *-iphone/Images/Default@2x.png:         640x960 iPad splash screen

- Android splash screens:

        cp ../spaceball-iphone/Images/Default\@2x.png res/drawable-hdpi/splash.png
        cp ../spaceball-iphone/Images/Default\@2x.png res/drawable-xhdpi/splash.png
        cp ../spaceball-iphone/Images/Default.png res/drawable-ldpi/splash.png
        cp ../spaceball-iphone/Images/Default.png res/drawable-mdpi/splash.png


- new_app_template.nsi:                   Windows Nullsoft Installer config
- new_app_template-android/*:             Android Eclipse linker and installer config
- new_app_template-iphone/*:              iPhone XCode binary signer config

- iphone-Info.plist:                      iPhone properties list
- mac-Info.plist:                         OSX properties list

