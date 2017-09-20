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

#ifndef LFL_ML_CART_H__
#define LFL_ML_CART_H__
namespace LFL {

/* classification and regression tree */
struct CART {
  struct Feature { int ind=0, count=0; string name; }; 

  struct FeatureSet {
    virtual int Size() const = 0;
    virtual Feature *Get(int ind) = 0;
  };

  struct Question {
    const char *name;
    virtual bool Answer(const Feature *) const = 0;
  };

  struct QuestionList {
    const char *name;
    virtual int Size() const = 0;
    virtual Question *Get(int ind) = 0;
  };

  /* classification */
  struct Tree {
    static const int map_buckets=5, map_values=2;
    QuestionList *inventory;
    unique_ptr<Matrix> questions;
    unique_ptr<vector<string>> name;
    HashMatrix namemap, leafnamemap;
  };

  static const double *Query(const Tree *model, const Feature *f) {
    string node="t";
    for (;;) {
      const double *he = model->namemap.Get(fnv32(node.c_str()));
      if (!he) break;

      int ind = he[1], qind = model->questions->row(ind)[0];
      const Question *q = model->inventory->Get(qind);
      node += q->Answer(f) ? "l" : "r";
    }

    return model->leafnamemap.Get(fnv32(node.c_str()));
  }

  static int Read(const char *dir, const char *name, int lastiter, QuestionList *inventory, Tree *out) {
    out->inventory = inventory;
    unique_ptr<Matrix> namemap_data, leafnamemap_data;
    if (MatrixFile::ReadVersioned(dir, name, "questions",   &out->questions,   0, lastiter)<0) return ERRORv(-1, name, ".", lastiter, ".questions"  );
    if (MatrixFile::ReadVersioned(dir, name, "namemap",     &namemap_data,     0, lastiter)<0) return ERRORv(-1, name, ".", lastiter, ".namemap"    );
    if (MatrixFile::ReadVersioned(dir, name, "leafnamemap", &leafnamemap_data, 0, lastiter)<0) return ERRORv(-1, name, ".", lastiter, ".leafnamemap");
    if (StringFile::ReadVersioned(dir, name, "name",        &out->name,        0, lastiter)<0) return ERRORv(-1, name, ".", lastiter, ".name"       );
    out->namemap     = HashMatrix(move(namemap_data),     Tree::map_values);
    out->leafnamemap = HashMatrix(move(leafnamemap_data), Tree::map_values);
    return 0;
  }

  static void Blank(QuestionList *inventory, Tree *out, double val) {
    out->inventory = inventory;
    out->namemap    .map->Open(4, Tree::map_buckets*Tree::map_values);
    out->leafnamemap.map->Open(4, Tree::map_buckets*Tree::map_values);
    double *he = out->leafnamemap.Set(fnv32("t"));
    he[1] = val;
  }

  /* regression */
  struct TreeBuilder {
    vector<string> name;
    vector<int> question;
    map<string, unsigned> leafname;
  };

  struct SplitRanker {
    virtual void Parent(FeatureSet *pool, const vector<int> &parent) = 0;
    virtual double Rank(FeatureSet *pool, const vector<int> &left, const vector<int> &right) = 0;
  };

  static void Split(FeatureSet *pool, Question *q, const vector<int> &parent, vector<int> &left, vector<int> &right) {
    for (int i=0; i<parent.size(); i++) {
      Feature *f = pool->Get(parent[i]);
      if (q->Answer(f)) left.push_back(f->ind);
      else             right.push_back(f->ind);
    }
  }

  static int BestSplit(FeatureSet *pool, QuestionList *Q, SplitRanker *ranker, const vector<int> &parent, int minsize) {
    ranker->Parent(pool, parent);

    double max; int maxind = -1;
    for (int i=0; i<Q->Size(); i++) {
      Question *q = Q->Get(i);

      vector<int> left, right;
      Split(pool, q, parent, left, right);

      int leftsum=0, rightsum=0;
      for (int j=0; j<left.size();  j++)  leftsum += pool->Get(left [j])->count;
      for (int j=0; j<right.size(); j++) rightsum += pool->Get(right[j])->count;
      if (leftsum<minsize || rightsum<minsize) continue;

      double rank = ranker->Rank(pool, left, right);
      if (maxind == -1 || rank > max) { max = rank; maxind = i; }
    }
    return maxind;
  }

  static void GrowTree(FeatureSet *pool, QuestionList *Q, SplitRanker *ranker, string name, const vector<int> &node, int stopnodesize, TreeBuilder &out) {
    if (node.size() < stopnodesize) { INFO(name, " done with ", node.size()); return AddLeaf(pool, name, node, out);  }

    int best_question = BestSplit(pool, Q, ranker, node, stopnodesize);
    if (best_question < 0) { INFO(name, " done because best_question = ", best_question); return AddLeaf(pool, name, node, out); }
    out.name.push_back(name);
    out.question.push_back(best_question);

    vector<int> left, right;
    Question *q = Q->Get(best_question);

    Split(pool, q, node, left, right);
    INFOf("%s %d best split by %s into %sl %d and %sr %d", name.c_str(), node.size(), q->name, name.c_str(), left.size(), name.c_str(), right.size());

    GrowTree(pool, Q, ranker, name + "l", left,  stopnodesize, out);
    GrowTree(pool, Q, ranker, name + "r", right, stopnodesize, out);
  }

  static void AddLeaf(FeatureSet *pool, string name, const vector<int> &node, TreeBuilder &out) {
    int max, maxind=-1;
    for (int i=0; i<node.size(); i++) {
      Feature *f = pool->Get(node[i]);
      if (maxind == -1 || f->count > max) { max = f->count; maxind = i; }
    }
    const char *leafname = pool->Get(node[maxind])->name.c_str();
    out.leafname[name] = fnv32(leafname);
    INFO(name, " named leaf ", leafname);
  }

  static int WriteTree(const TreeBuilder &tree, QuestionList *QL, const char *name, const char *dir, int iteration) {
    if (tree.name.size() != tree.question.size()) { ERROR("invalid size ", tree.name.size(), " ", tree.question.size()); return -1; }
    const char *flagtext = QL->name;

    int buckets=Tree::map_buckets, values=Tree::map_values;
    Matrix namemap_data    (NextPrime(tree.name.size()*4),     buckets*values);
    Matrix leafnamemap_data(NextPrime(tree.leafname.size()*4), buckets*values);
    HashMatrix namemap(&namemap_data, values), leafnamemap(&leafnamemap_data, values);

    LocalFile names    (string(dir) + MatrixFile::Filename(name, "name",      "string", iteration), "w");
    LocalFile questions(string(dir) + MatrixFile::Filename(name, "questions", "matrix", iteration), "w");

    MatrixFile::WriteHeader(&names,     BaseName(names.Filename()),     flagtext, tree.name.size(),     1);
    MatrixFile::WriteHeader(&questions, BaseName(questions.Filename()), flagtext, tree.question.size(), 1);

    for (int i=0; i<tree.question.size(); i++) {
      const char *n = tree.name[i].c_str();
      StringFile::WriteRow(&names, n);

      double qrow[1] = { (double)tree.question[i] };
      MatrixFile::WriteRow(&questions, qrow, 1);

      double *he = namemap.Set(fnv32(n));
      if (!he) FATAL("Matrix hash collision: ", n);
      he[1] = i;
    }

    for (map<string, unsigned>::const_iterator i = tree.leafname.begin(); i != tree.leafname.end(); i++) {
      const char *n = (*i).first.c_str();
      unsigned v = (*i).second;
      double *he = leafnamemap.Set(fnv32(n));
      if (!he) FATAL("Matrix hash collision: ", n);
      he[1] = v;
    }

    if (MatrixFile(namemap.map.get(),     flagtext).WriteVersioned(dir, name,     "namemap", iteration)<0) { ERROR(name, " write namemap"    ); return -1; }
    if (MatrixFile(leafnamemap.map.get(), flagtext).WriteVersioned(dir, name, "leafnamemap", iteration)<0) { ERROR(name, " write leafnamemap"); return -1; }

    return 0;
  }
};

struct PhoneticDecisionTree {
  struct Feature : public CART::Feature { int D=0; double *mean=0, *covar=0; };

  struct FeatureSet : public CART::FeatureSet {
    int N;
    vector<Feature> S;
    FeatureSet(int n) : N(n), S(N, Feature()) {}

    int Size() const { return N; }
    CART::Feature *Get(int ind) { return (ind>=0 && ind<N) ? &S[ind] : 0; }
  };

  struct Question : public CART::Question {
    int is_phone;
    vector<int> member;

    bool Answer(const CART::Feature *feat) const {
      Feature *f = (Feature *)feat;
      int prevPhone, nextPhone, phone=AcousticModel::ParseName(f->name, 0, &prevPhone, &nextPhone);
      if (phone < 0) ERROR("AcousticModel::parseName failed for ", f->name);

      int question_refers = is_phone < 0 ? prevPhone : nextPhone;
      for (int i=0; i<member.size(); i++) if (member[i] == question_refers) return true;
      return false;
    }
  };

  struct QuestionList : public CART::QuestionList {
    int N;
    vector<Question> L;
    QuestionList(int n) : N(n), L(N, Question()) { name="PhoneticDecisionTree::QuestionList"; }

    int Size() const { return N; }
    CART::Question *Get(int ind) { return (ind>=0 && ind<N) ? &L[ind] : 0; }

    static unique_ptr<QuestionList> Create() {
      int count=0;
#     undef  XX
#     define XX(x) count++;
#     undef  YY
#     define YY(x)
#     include "core/speech/phonics.h"

      auto QL = make_unique<QuestionList>(count);
      Question *q;
      int phone=-1, ind=0;
#     undef  XX
#     define XX(x) q = &QL->L[ind++]; q->name = #x; q->is_phone = phone;
#     undef  YY
#     define YY(x) q->member.push_back(Phoneme::x);
#     include "core/speech/phonics.h"

      phone=1; ind=0;
#     include "core/speech/phonics.h"

      return QL;
    }
  };

  struct SplitRanker : public CART::SplitRanker {
    double parent_likelihood;

    void Parent(CART::FeatureSet *pool, const vector<int> &node) { parent_likelihood = Likelihood(pool, node); }

    double Rank(CART::FeatureSet *pool, const vector<int> &left, const vector<int> &right) {
      double left_likelihood = Likelihood(pool, left), right_likelihood = Likelihood(pool, right);
      return left_likelihood + right_likelihood - parent_likelihood;
    }

    static double Likelihood(CART::FeatureSet *pool, const vector<int> &node) {
      int D = ((Feature *)pool->Get(0))->D;
      vector<double> mean(D, 0), covar(D, 0), x(D, 0);
      double denom = 0;

      /* determine node mean */
      for (int i=0; i<node.size(); i++) {
        Feature *f = (Feature *)pool->Get(node[i]);

        Vector::Assign(&x[0], f->mean, D);
        Vector::Mult(&x[0], f->count, D);
        Vector::Add(&mean[0], x.data(), D);

        denom += f->count;
      }
      Vector::Div(&mean[0], denom, D);

      /* determine node covariance */
      denom = 0;
      for (int i=0; i<node.size(); i++) {
        Feature *f = (Feature *)pool->Get(node[i]);

        Vector::Assign(&x[0], f->mean, D);
        Vector::Mult(&x[0], x.data(), D);
        Vector::Add(&x[0], f->covar, D);
        Vector::Mult(&x[0], f->count, D);
        Vector::Add(&covar[0], x.data(), D);

        denom += f->count;
      }
      Vector::Div(&covar[0], denom, D);

      Vector::Assign(&x[0], mean.data(), D);
      Vector::Mult(&x[0], x.data(), D);

      Vector::Sub(&covar[0], x.data(), D);

      /* determine node likelihood */ 
      return -0.5 * ((1 + log(2*M_PI))*D + DiagDet(covar.data(), D)) * denom;
    }
  };
};

}; // namespace LFL
#endif // LFL_ML_CART_H__
