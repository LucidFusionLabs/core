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

#ifndef LFL_CORE_APP_TOOLKIT_H__
#define LFL_CORE_APP_TOOLKIT_H__
namespace LFL {

struct MenuItem { string shortcut, name; Callback cb; int image; };
struct AlertItem { string first, second; StringCB cb; int image; };
struct PanelItem { string type; Box box; StringCB cb; int image; };

struct PickerItem {
  typedef function<bool(PickerItem*)> CB;
  vector<vector<string>> data;
  vector<int> picked;
  CB cb;
  string Picked(int i) const { return data[i][picked[i]]; }
  string PickedString() const
  { string v; for (int i=0, l=data.size(); i!=l; ++i) StrAppend(&v, i ? " " : "", Picked(i)); return v; }
};

struct TableItem {
  enum { None=0, Label=1, Separator=2, Command=3, Button=4, Toggle=5, Selector=6, Picker=7, Slider=8,
    TextInput=9, NumberInput=10, PasswordInput=11, FontPicker=12 }; 
  struct Flag { enum { LeftText=1, SubText=2, FixDropdown=4, HideKey=8, PlaceHolderVal=16, ColoredSubText=32,
    ColoredRightText=64, User1=128 }; };
  string key, val, right_text, dropdown_key;
  int type, tag, flags, left_icon, right_icon, selected=0, height=0;
  Callback cb;
  StringCB right_cb;
  PickerItem *picker;
  bool hidden;
  unsigned char fg_r=0, fg_g=0, fg_b=0, fg_a=0;
  unsigned char bg_r=0, bg_g=0, bg_b=0, bg_a=0;
  float minval=0, maxval=0;
  virtual ~TableItem() {}
  TableItem(string K=string(), int T=0, string V=string(), string RT=string(), int TG=0, int LI=0, int RI=0,
            Callback CB=Callback(), StringCB RC=StringCB(), int F=0, bool H=false, PickerItem *P=0, string DDK=string()) :
    key(move(K)), val(move(V)), right_text(move(RT)), dropdown_key(move(DDK)), type(T), tag(TG), flags(F),
    left_icon(LI), right_icon(RI), cb(move(CB)), right_cb(move(RC)), picker(P), hidden(H) {}
  TableItem(string K, int T, string V, string RT, int TG, int LI, int RI, Callback CB, StringCB RC, int F,
            bool H, PickerItem *P, string DDK, const Color &fg, const Color &bg, float MinV=0, float MaxV=0);
  bool HasPlaceholderValue() const { return val.size() && (val[0] == 1 || val[0] == 2); }
  string GetPlaceholderValue() const { return val.substr(1); }
  void CheckAssign(const string &k, Callback c) { CHECK_EQ(k, key); cb=move(c); }
  void SetFGColor(const Color&);
  void SetBGColor(const Color&);
};

struct TableSectionInterface {
  struct Flag { enum { EditButton=1, EditableIfHasTag=2, MovableRows=4, DoubleRowHeight=8,
    HighlightSelectedRow=16, DeleteRowsWhenAllHidden=32, ClearLeftNavWhenEmpty=64, User1=128 }; };
  struct Change { int section, row; string val; bool hidden; int left_icon, right_icon, type; string key; Callback cb; int flags; };
  typedef vector<Change> ChangeList;
  typedef unordered_map<string, ChangeList> ChangeSet;
  virtual ~TableSectionInterface() {}
  static void ApplyChange(TableItem *out, const Change &d);
};

template <class TI> struct TableSection : public TableSectionInterface {
  int flag=0, header_height=0, start_row=0, editable_startrow=-1, editable_skiplastrows=0;
  IntIntCB delete_row_cb;
  TI header;
  vector<TI> item;
  TableSection(int sr=0) : start_row(sr) {}
  TableSection(TI h, int f=0, int sr=0) : flag(f), start_row(sr), header(move(h)) {}
  void SetEditable(int sr, int sll, IntIntCB cb) { editable_startrow=sr; editable_skiplastrows=sll; delete_row_cb=move(cb); }

  static vector<TableSection> Convert(vector<TableItem> in) {
    vector<TableSection> ret;
    ret.emplace_back();
    for (auto &i : in) {
      if (i.type == TableItem::Separator) ret.emplace_back(move(i));
      else                                ret.back().item.emplace_back(move(i));
    }
    return ret;
  }

  static void FindSectionOffset(const vector<TableSection> &data, int collapsed_row, int *section_out, int *row_out) {
    auto it = lower_bound(data.begin(), data.end(), TableSection(collapsed_row),
                          MemberLessThanCompare<TableSection, int, &TableSection::start_row>());
    if (it != data.end() && it->start_row == collapsed_row) { *section_out = it - data.begin(); return; }
    CHECK_NE(data.begin(), it);
    *section_out = (it != data.end() ? (it - data.begin()) : data.size()) - 1;
    *row_out = collapsed_row - data[*section_out].start_row - 1;
  }

  static void ApplyChangeList(const ChangeList &changes, vector<TableSection> *out, function<void(const Change&)> f) {
    for (auto &d : changes) {
      CHECK_LT(d.section, out->size());
      CHECK_LT(d.row, (*out)[d.section].item.size());
      auto &ci = (*out)[d.section].item[d.row];
      ApplyChange(&ci, d);
      if (f) f(d);
    }
  }
};

typedef vector<MenuItem>  MenuItemVec;
typedef vector<AlertItem> AlertItemVec;
typedef vector<PanelItem> PanelItemVec;
typedef vector<TableItem> TableItemVec;

struct TimerInterface {
  virtual ~TimerInterface() {}
  virtual bool Clear() = 0;
  virtual void Run(Time interval, bool force=false) = 0;
};

struct VideoResamplerInterface {
  int s_fmt=0, d_fmt=0, s_width=0, d_width=0, s_height=0, d_height=0;
  virtual ~VideoResamplerInterface() {}
  virtual bool Opened() const = 0;
  virtual void Open(int sw, int sh, int sf, int dw, int dh, int df) = 0;
  virtual void Resample(const unsigned char *s, int sls, unsigned char *d, int dls, bool flip_x=0, bool flip_y=0) = 0;
};

struct AlertViewInterface {
  virtual ~AlertViewInterface() {}
  virtual void Hide() = 0;
  virtual void Show(const string &arg) = 0;
  virtual void ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) = 0;
  virtual string RunModal(const string &arg) = 0;
};

struct PanelViewInterface {
  virtual ~PanelViewInterface() {}
  virtual void Show() = 0;
  virtual void SetTitle(const string &title) = 0;
};

struct ToolbarViewInterface {
  enum { BORDERLESS_BUTTONS=1 };
  virtual ~ToolbarViewInterface() {}
  virtual void Show(bool show_or_hide) = 0;
  virtual void ToggleButton(const string &n) = 0;
  virtual void SetTheme(const string &theme) = 0;
  virtual string GetTheme() = 0;
};

struct MenuViewInterface {
  virtual ~MenuViewInterface() {}
  virtual void Show() = 0;
};

struct TableViewInterface {
  bool changed=0;
  Callback hide_cb, show_cb = [=](){ changed=0; }; 
  virtual ~TableViewInterface() {}

  virtual void DelNavigationButton(int id) = 0;
  virtual void AddNavigationButton(int id, const TableItem &item) = 0;
  virtual void SetToolbar(ToolbarViewInterface*) = 0;
  virtual void Show(bool show_or_hide) = 0;

  virtual string GetKey(int section, int row) = 0;
  virtual string GetValue(int section, int row) = 0;
  virtual int GetTag(int section, int row) = 0;
  virtual PickerItem *GetPicker(int section, int row) = 0;
  virtual StringPairVec GetSectionText(int section) = 0;

  virtual void BeginUpdates() = 0;
  virtual void EndUpdates() = 0;
  virtual void AddRow(int section, TableItem item) = 0;
  virtual void SelectRow(int section, int row) = 0;
  virtual void ReplaceRow(int section, int row, TableItem item) = 0;
  virtual void ReplaceSection(int section, TableItem header, int flag, TableItemVec item) = 0;
  virtual void ApplyChangeList(const TableSectionInterface::ChangeList&) = 0;
  virtual void SetSectionValues(int section, const StringVec&) = 0;
  virtual void SetSectionColors(int seciton, const vector<Color>&) = 0;
  virtual void SetSectionEditable(int section, int start_row, int skip_last_rows, IntIntCB cb=IntIntCB()) = 0;
  virtual void SetHeader(int section, TableItem header) = 0;
  virtual void SetKey(int secton, int row, const string &key) = 0;
  virtual void SetTag(int section, int row, int val) = 0;
  virtual void SetValue(int section, int row, const string &val) = 0;
  virtual void SetSelected(int section, int row, int selected) = 0;
  virtual void SetHidden(int section, int row, int val) = 0;
  virtual void SetColor(int section, int row, const Color &val) = 0;
  virtual void SetTitle(const string &title) = 0;
  virtual void SetTheme(const string &theme) = 0;

  bool GetSectionText(int section, vector<string*> out, bool check=1) { return GetPairValues(GetSectionText(section), move(out), check); }
  void ApplyChangeSet(const string &v, const TableSectionInterface::ChangeSet &changes);
};

struct TableViewController {
  unique_ptr<TableViewInterface> view;
  TableViewController() {}
  TableViewController(unique_ptr<TableViewInterface> v) : view(move(v)) {}
  virtual ~TableViewController() {}
};

struct TextViewInterface {
  Callback hide_cb, show_cb;
  virtual ~TextViewInterface() {}
};

struct NavigationViewInterface {
  bool shown=0;
  TableViewInterface *root=0;
  virtual ~NavigationViewInterface() {}
  virtual TableViewInterface *Back() = 0;
  virtual void Show(bool show_or_hide) = 0;
  virtual void PushTableView(TableViewInterface*) = 0;
  virtual void PushTextView(TextViewInterface*) = 0;
  virtual void PopView(int num=1) = 0;
  virtual void PopToRoot() = 0;
  virtual void PopAll() = 0;
  virtual void SetTheme(const string &theme) = 0;
};

struct AdvertisingViewInterface {
  struct Type { enum { BANNER=1 }; };
  virtual ~AdvertisingViewInterface() {}
  virtual void Show(bool show_or_hide) = 0;
  virtual void Show(TableViewInterface*, bool show_or_hide) = 0;
};

struct ProductInterface {
  string id;
  ProductInterface(const string &i) : id(i) {}
  virtual ~ProductInterface() {}
  virtual string Name() = 0;
  virtual string Description() = 0;
  virtual string Price() = 0;
};

struct PurchasesInterface {
  typedef function<void(unique_ptr<ProductInterface>)> ProductCB;
  virtual ~PurchasesInterface() {}
  virtual bool CanPurchase() = 0;
  virtual void LoadPurchases() = 0;
  virtual bool HavePurchase(const string&) = 0;
  virtual void RestorePurchases(Callback done_cb) = 0;
  virtual void PreparePurchase(const StringVec&, Callback done_cb, ProductCB product_cb) = 0;
  virtual bool MakePurchase(ProductInterface*, IntCB result_cb) = 0;
};

struct NagInterface {
  virtual ~NagInterface() {}
};

struct ToolkitInterface {
  virtual unique_ptr<AlertViewInterface> CreateAlert(AlertItemVec items) = 0;
  virtual unique_ptr<PanelViewInterface> CreatePanel(const Box&, const string &title, PanelItemVec) = 0;
  virtual unique_ptr<ToolbarViewInterface> CreateToolbar(const string &theme, MenuItemVec items, int flag) = 0;
  virtual unique_ptr<MenuViewInterface> CreateMenu(const string &title, MenuItemVec items) = 0;
  virtual unique_ptr<MenuViewInterface> CreateEditMenu(MenuItemVec items) = 0;
  virtual unique_ptr<TableViewInterface> CreateTableView
    (const string &title, const string &style, const string &theme, TableItemVec items) = 0;
  virtual unique_ptr<TextViewInterface> CreateTextView(const string &title, File *file) = 0;
  virtual unique_ptr<TextViewInterface> CreateTextView(const string &title, const string &text) = 0;
  virtual unique_ptr<NavigationViewInterface> CreateNavigationView(const string &style, const string &theme) = 0;
};

struct SystemToolkit : public ToolkitInterface {
  unique_ptr<AlertViewInterface> CreateAlert(AlertItemVec items);
  unique_ptr<PanelViewInterface> CreatePanel(const Box&, const string &title, PanelItemVec);
  unique_ptr<ToolbarViewInterface> CreateToolbar(const string &theme, MenuItemVec items, int flag);
  unique_ptr<MenuViewInterface> CreateMenu(const string &title, MenuItemVec items);
  unique_ptr<MenuViewInterface> CreateEditMenu(MenuItemVec items);
  unique_ptr<TableViewInterface> CreateTableView
    (const string &title, const string &style, const string &theme, TableItemVec items);
  unique_ptr<TextViewInterface> CreateTextView(const string &title, File *file);
  unique_ptr<TextViewInterface> CreateTextView(const string &title, const string &text);
  unique_ptr<NavigationViewInterface> CreateNavigationView(const string &style, const string &theme);

  static void DisableAdvertisingCrashReporting();
  static unique_ptr<TimerInterface> CreateTimer(Callback cb);
  static unique_ptr<AdvertisingViewInterface> CreateAdvertisingView
    (int type, int placement, const string &id, const StringVec &test_devices);
  static unique_ptr<PurchasesInterface> CreatePurchases(string);
  static unique_ptr<NagInterface> CreateNag(const string &id, int min_days, int min_uses, int min_events, int remind_days);
};

}; // namespace LFL
#endif // LFL_CORE_APP_GUI_TOOLKIT_H__
