#!/usr/bin/ruby

require 'xcodeproj'

project_path = 'Project.xcodeproj'
configurations = ['Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel']

project = Xcodeproj::Project.open(project_path)
project.targets.each do |target|
  configurations.each do |configuration|  
    settings = target.build_settings(configuration)
    if (settings['GCC_PREPROCESSOR_DEFINITIONS'] != nil)
        settings['GCC_PREPROCESSOR_DEFINITIONS'] << '$(inherited)'
    end
    if (settings['HEADER_SEARCH_PATHS'] != nil)
        settings['HEADER_SEARCH_PATHS'] << '$(inherited)'
    end
    if (settings['LIBRARY_SEARCH_PATHS'] != nil)
        settings['LIBRARY_SEARCH_PATHS'] << '$(inherited)'
    end
    if (settings['OTHER_LDFLAGS'] != nil)
        settings['OTHER_LDFLAGS'] << '$(inherited)'
    end
  end
end

project.save
