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

#ifdef LFL_C_SYNTAX_STATEMENT
XX(goto)
XX(break)
XX(return)
XX(continue)
XX(asm)
XX(__asm__)
#endif

#ifdef LFL_C_SYNTAX_STRUCTURE
XX(struct)
XX(union)
XX(enum)
XX(typedef)
#endif

#ifdef LFL_C_SYNTAX_LABEL
XX(case)
XX(default)
#endif

#ifdef LFL_C_SYNTAX_CONDITIONAL
XX(if)
XX(else)
XX(switch)
#endif

#ifdef LFL_C_SYNTAX_REPEAT
XX(while)
XX(for)
XX(do)
#endif

#ifdef LFL_C_SYNTAX_TODO
XX(contained)
XX(TODO)
XX(FIXME)
XX(XXX)
#endif

#ifdef LFL_C_SYNTAX_OPERATOR
XX(sizeof)
XX(typeof)
XX(__real__)
XX(__imag__)
#endif

#ifdef LFL_C_SYNTAX_TYPE
XX(int)
XX(long)
XX(short)
XX(char)
XX(void)
XX(signed)
XX(unsigned)
XX(float)
XX(double)
XX(size_t)
XX(ssize_t)
XX(off_t)
XX(wchar_t)
XX(ptrdiff_t)
XX(sig_atomic_t)
XX(fpos_t)
XX(clock_t)
XX(time_t)
XX(va_list)
XX(jmp_buf)
XX(FILE)
XX(DIR)
XX(div_t)
XX(ldiv_t)
XX(mbstate_t)
XX(wctrans_t)
XX(wint_t)
XX(wctype_t)
XX(bool)
XX(complex)
XX(int8_t)
XX(int16_t)
XX(int32_t)
XX(int64_t)
XX(uint8_t)
XX(uint16_t)
XX(uint32_t)
XX(uint64_t)
XX(int_least8_t)
XX(int_least16_t)
XX(int_least32_t)
XX(int_least64_t)
XX(uint_least8_t)
XX(uint_least16_t)
XX(uint_least32_t)
XX(uint_least64_t)
XX(int_fast8_t)
XX(int_fast16_t)
XX(int_fast32_t)
XX(int_fast64_t)
XX(uint_fast8_t)
XX(uint_fast16_t)
XX(uint_fast32_t)
XX(uint_fast64_t)
XX(intptr_t)
XX(uintptr_t)
XX(intmax_t)
XX(uintmax_t)
XX(__label__)
XX(__complex__)
XX(__volatile__)
#endif

#ifdef LFL_C_SYNTAX_STORAGECLASS
XX(static)
XX(register)
XX(auto)
XX(volatile)
XX(extern)
XX(const)
XX(inline)
XX(__attribute__)
XX(restrict)
#endif

#ifdef LFL_C_SYNTAX_CONSTANT
XX(__GNUC__)
XX(__FUNCTION__)
XX(__PRETTY_FUNCTION__)
XX(__func__)
XX(__LINE__)
XX(__FILE__)
XX(__DATE__)
XX(__TIME__)
XX(__STDC__)
XX(__STDC_VERSION__)
XX(CHAR_BIT)
XX(MB_LEN_MAX)
XX(MB_CUR_MAX)
XX(UCHAR_MAX)
XX(UINT_MAX)
XX(ULONG_MAX)
XX(USHRT_MAX)
XX(CHAR_MIN)
XX(INT_MIN)
XX(LONG_MIN)
XX(SHRT_MIN)
XX(CHAR_MAX)
XX(INT_MAX)
XX(LONG_MAX)
XX(SHRT_MAX)
XX(SCHAR_MIN)
XX(SINT_MIN)
XX(SLONG_MIN)
XX(SSHRT_MIN)
XX(SCHAR_MAX)
XX(SINT_MAX)
XX(SLONG_MAX)
XX(SSHRT_MAX)
XX(__func__)
XX(LLONG_MIN)
XX(LLONG_MAX)
XX(ULLONG_MAX)
XX(INT8_MIN)
XX(INT16_MIN)
XX(INT32_MIN)
XX(INT64_MIN)
XX(INT8_MAX)
XX(INT16_MAX)
XX(INT32_MAX)
XX(INT64_MAX)
XX(UINT8_MAX)
XX(UINT16_MAX)
XX(UINT32_MAX)
XX(UINT64_MAX)
XX(INT_LEAST8_MIN)
XX(INT_LEAST16_MIN)
XX(INT_LEAST32_MIN)
XX(INT_LEAST64_MIN)
XX(INT_LEAST8_MAX)
XX(INT_LEAST16_MAX)
XX(INT_LEAST32_MAX)
XX(INT_LEAST64_MAX)
XX(UINT_LEAST8_MAX)
XX(UINT_LEAST16_MAX)
XX(UINT_LEAST32_MAX)
XX(UINT_LEAST64_MAX)
XX(INT_FAST8_MIN)
XX(INT_FAST16_MIN)
XX(INT_FAST32_MIN)
XX(INT_FAST64_MIN)
XX(INT_FAST8_MAX)
XX(INT_FAST16_MAX)
XX(INT_FAST32_MAX)
XX(INT_FAST64_MAX)
XX(UINT_FAST8_MAX)
XX(UINT_FAST16_MAX)
XX(UINT_FAST32_MAX)
XX(UINT_FAST64_MAX)
XX(INTPTR_MIN)
XX(INTPTR_MAX)
XX(UINTPTR_MAX)
XX(INTMAX_MIN)
XX(INTMAX_MAX)
XX(UINTMAX_MAX)
XX(PTRDIFF_MIN)
XX(PTRDIFF_MAX)
XX(SIG_ATOMIC_MIN)
XX(SIG_ATOMIC_MAX)
XX(SIZE_MAX)
XX(WCHAR_MIN)
XX(WCHAR_MAX)
XX(WINT_MIN)
XX(WINT_MAX)
XX(FLT_RADIX)
XX(FLT_ROUNDS)
XX(FLT_DIG)
XX(FLT_MANT_DIG)
XX(FLT_EPSILON)
XX(DBL_DIG)
XX(DBL_MANT_DIG)
XX(DBL_EPSILON)
XX(LDBL_DIG)
XX(LDBL_MANT_DIG)
XX(LDBL_EPSILON)
XX(FLT_MIN)
XX(FLT_MAX)
XX(FLT_MIN_EXP)
XX(FLT_MAX_EXP)
XX(FLT_MIN_10_EXP)
XX(FLT_MAX_10_EXP)
XX(DBL_MIN)
XX(DBL_MAX)
XX(DBL_MIN_EXP)
XX(DBL_MAX_EXP)
XX(DBL_MIN_10_EXP)
XX(DBL_MAX_10_EXP)
XX(LDBL_MIN)
XX(LDBL_MAX)
XX(LDBL_MIN_EXP)
XX(LDBL_MAX_EXP)
XX(LDBL_MIN_10_EXP)
XX(LDBL_MAX_10_EXP)
XX(HUGE_VAL)
XX(CLOCKS_PER_SEC)
XX(NULL)
XX(LC_ALL)
XX(LC_COLLATE)
XX(LC_CTYPE)
XX(LC_MONETARY)
XX(LC_NUMERIC)
XX(LC_TIME)
XX(SIG_DFL)
XX(SIG_ERR)
XX(SIG_IGN)
XX(SIGABRT)
XX(SIGFPE)
XX(SIGILL)
XX(SIGHUP)
XX(SIGINT)
XX(SIGSEGV)
XX(SIGTERM)
XX(SIGABRT)
XX(SIGALRM)
XX(SIGCHLD)
XX(SIGCONT)
XX(SIGFPE)
XX(SIGHUP)
XX(SIGILL)
XX(SIGINT)
XX(SIGKILL)
XX(SIGPIPE)
XX(SIGQUIT)
XX(SIGSEGV)
XX(SIGSTOP)
XX(SIGTERM)
XX(SIGTRAP)
XX(SIGTSTP)
XX(SIGTTIN)
XX(SIGTTOU)
XX(SIGUSR1)
XX(SIGUSR2)
XX(_IOFBF)
XX(_IOLBF)
XX(_IONBF)
XX(BUFSIZ)
XX(EOF)
XX(WEOF)
XX(FOPEN_MAX)
XX(FILENAME_MAX)
XX(L_tmpnam)
XX(SEEK_CUR)
XX(SEEK_END)
XX(SEEK_SET)
XX(TMP_MAX)
XX(stderr)
XX(stdin)
XX(stdout)
XX(EXIT_FAILURE)
XX(EXIT_SUCCESS)
XX(RAND_MAX)
XX(E2BIG)
XX(EACCES)
XX(EAGAIN)
XX(EBADF)
XX(EBADMSG)
XX(EBUSY)
XX(ECANCELED)
XX(ECHILD)
XX(EDEADLK)
XX(EDOM)
XX(EEXIST)
XX(EFAULT)
XX(EFBIG)
XX(EILSEQ)
XX(EINPROGRESS)
XX(EINTR)
XX(EINVAL)
XX(EIO)
XX(EISDIR)
XX(EMFILE)
XX(EMLINK)
XX(EMSGSIZE)
XX(ENAMETOOLONG)
XX(ENFILE)
XX(ENODEV)
XX(ENOENT)
XX(ENOEXEC)
XX(ENOLCK)
XX(ENOMEM)
XX(ENOSPC)
XX(ENOSYS)
XX(ENOTDIR)
XX(ENOTEMPTY)
XX(ENOTSUP)
XX(ENOTTY)
XX(ENXIO)
XX(EPERM)
XX(EPIPE)
XX(ERANGE)
XX(EROFS)
XX(ESPIPE)
XX(ESRCH)
XX(ETIMEDOUT)
XX(EXDEV)
XX(M_E)
XX(M_LOG2E)
XX(M_LOG10E)
XX(M_LN2)
XX(M_LN10)
XX(M_PI)
XX(M_PI_2)
XX(M_PI_4)
XX(M_1_PI)
XX(M_2_PI)
XX(M_2_SQRTPI)
XX(M_SQRT2)
XX(M_SQRT1_2)
XX(true)
XX(false)
#endif
