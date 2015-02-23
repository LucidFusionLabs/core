// -*- C++ -*-

class CRegExpException
{
public:
    CRegExpException(int nError);

    int GetError() const;
    CString GetErrorString() const;

private:
    int m_nError;
};

class CRegExp
{
public:
    CRegExp(LPCTSTR pszPattern);
    ~CRegExp();

    BOOL Exec(const CString& sMatch);
    BOOL IsMatched(int nSubExp = 0) const;
    int GetMatchStart(int nSubExp = 0) const;
    int GetMatchEnd(int nSubExp = 0) const;
    CString GetMatch(int nSubExp = 0) const;
	int GetNumberOfMatches() const;

protected:
    void ResetMatches();
    void ConvertError(int nErrorCode) const;
    
    typedef regexp *PREGEXP;
    PREGEXP m_preCompiled;
    CString m_sMatch;
    std::vector<regmatch> m_arMatches;

private:
    // disable copying
    CRegExp(const CRegExp&);
    CRegExp& operator=(const CRegExp&);
};

