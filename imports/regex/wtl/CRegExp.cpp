#include "stdafx.h"

#include <vector>
// include the C regexp code
#ifdef _UNICODE

#define REGEXP_UNICODE
#define re_comp_t re_comp_w
#define re_exec_t re_exec_w

#else

#undef REGEXP_UNICODE
#define re_comp_t re_comp
#define re_exec_t re_exec

#endif
#include "regexp.h"

#include "CRegExp.h"


CRegExpException::CRegExpException(int nError) :
    m_nError(nError)
{}

int CRegExpException::GetError() const
{
    return m_nError;
}

CString CRegExpException::GetErrorString() const
{
    char arTemp[128];
    re_error(m_nError, NULL, arTemp, sizeof(arTemp));
    CString retval(arTemp);
    return retval;
}

CRegExp::CRegExp(LPCTSTR pszPattern) :
    m_preCompiled(NULL)
{
    int nErrCode = re_comp_t(&m_preCompiled, pszPattern);
    ConvertError(nErrCode);

    try
    {
        nErrCode = re_nsubexp(m_preCompiled);
        ConvertError(nErrCode);
        
        m_arMatches.resize(nErrCode);
        ResetMatches();
    }
    catch(...)
    {
        re_free(m_preCompiled);
        throw;
    }
}

CRegExp::~CRegExp()
{
    re_free(m_preCompiled);
}

BOOL CRegExp::Exec(const CString& sMatch)
{
    m_sMatch = sMatch;
    ResetMatches();
    int nErrCode = re_exec_t(m_preCompiled,
                             (LPCTSTR)m_sMatch,
                             m_arMatches.size(),
                             &m_arMatches[0]);
    ConvertError(nErrCode);
    return nErrCode > 0 ? TRUE : FALSE;
}

BOOL CRegExp::IsMatched(int nSubExp) const
{
    return (nSubExp < m_arMatches.size() &&
            m_arMatches[nSubExp].begin != -1) ?
        TRUE : FALSE;
}

int CRegExp::GetMatchStart(int nSubExp) const
{
    return nSubExp >= m_arMatches.size() ?
        -1 :
        m_arMatches[nSubExp].begin;
}

int CRegExp::GetMatchEnd(int nSubExp) const
{
    return nSubExp >= m_arMatches.size() ?
        -1 :
        m_arMatches[nSubExp].end;
}

CString CRegExp::GetMatch(int nSubExp) const
{
    if(nSubExp >= m_arMatches.size())
        return CString();

    regmatch rmMatch = m_arMatches[nSubExp];
    
    if(rmMatch.begin == -1 || rmMatch.end == -1)
        return CString();

    return m_sMatch.Mid(rmMatch.begin,
                        rmMatch.end - rmMatch.begin);
}

int CRegExp::GetNumberOfMatches() const
{
	return m_arMatches.size();
}

void CRegExp::ResetMatches()
{
    regmatch rmDummy;
    rmDummy.begin = rmDummy.end = -1;
    int nSize = m_arMatches.size();

    for(int nIndex = 0; nIndex < nSize; ++nIndex)
        m_arMatches[nIndex] = rmDummy;
}

void CRegExp::ConvertError(int nErrorCode) const
{
    if(nErrorCode < 0)
        throw CRegExpException(nErrorCode);
}

