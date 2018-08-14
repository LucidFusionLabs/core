/*
 * $Id$
 * Copyright (C) 2009 Lucid Fusion Labs

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LFL_ML_CORPUS_H__
#define LFL_ML_CORPUS_H__
namespace LFL {

struct Corpus {
  Callback start_cb, finish_cb;
  virtual ~Corpus() {}
  virtual void RunFile(const string &filename) {}
  virtual void Run(FileSystem *fs, const string &file_or_dir, ApplicationLifetime *lifetime=0) {
    if (start_cb) start_cb();
    if (!file_or_dir.empty() && !fs->IsDirectory(file_or_dir)) RunFile(file_or_dir);
    else {
      auto iter = fs->ReadDirectory(file_or_dir, -1);
      for (auto fn = iter->Next(); (!lifetime || lifetime->run) && fn; fn = iter->Next()) 
        Run(fs, StrCat(file_or_dir, fn), lifetime);
    }
    if (finish_cb) finish_cb();
  }  
};  

}; // namespace LFL
#endif // LFL_ML_CORPUS_H__
