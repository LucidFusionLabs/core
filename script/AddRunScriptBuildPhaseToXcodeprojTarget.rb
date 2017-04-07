#!/usr/bin/ruby

require 'xcodeproj'

project = Xcodeproj::Project.open(ARGV[0])
project.targets.each do |target|
  if (target.name == ARGV[1])
    phase = target.new_shell_script_build_phase("Run Script")
    phase.shell_script = ARGV[2]
  end
end
project.save()
