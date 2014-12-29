/*
 * $Id: corpus.h 1306 2014-09-04 07:13:16Z justin $
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

#ifndef __LFL_ML_CORPUS_H__
#define __LFL_ML_CORPUS_H__
namespace LFL {

struct Corpus {
    virtual ~Corpus() {}
    virtual void RunBegin(void *cb, void *arg) {} 
    virtual void Run(const char *filename, void *cb, void *arg) = 0;

    static void Run(Corpus *runner, const char *file_or_dir, void *cb, void *arg) {
        if (runner) runner->RunBegin(cb, arg);
        if (!file_or_dir || !*file_or_dir) return;
        Run(runner, file_or_dir, cb, arg, 0);
    }

    static void Run(Corpus *runner, const char *file_or_dir, void *cb, void *arg, void *arg2) {
        if (!LocalFile::IsDirectory(file_or_dir))
            return runner->Run(file_or_dir, cb, arg);                     

        DirectoryIter iter(file_or_dir, -1);
        for (const char *fn = iter.Next(); Running() && fn; fn = iter.Next()) {
            string pn = StrCat(file_or_dir, fn);
            Run(runner, pn.c_str(), cb, arg, arg2);
        }
    }  
};  

}; // namespace LFL
#endif // __LFL_ML_CORPUS_H__
