// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __STDC_WANT_LIB_EXT1__
#define __STDC_WANT_LIB_EXT1__ 1
#endif

#include "util/common.h"
#include <string.h>

#if (SIGMA_SYSTEM == SIGMA_SYSTEM_WINDOWS)
#include <Windows.h>
#endif

using namespace std;

namespace sigma
{
    namespace util
    {
        void sigma_memzero(void *data, size_t size)
        {
#if (SIGMA_SYSTEM == SIGMA_SYSTEM_WINDOWS)
            SecureZeroMemory(data, size);
#elif defined(SIGMA_USE_MEMSET_S)
            if (size > 0U && memset_s(data, static_cast<rsize_t>(size), 0, static_cast<rsize_t>(size)) != 0)
            {
                throw runtime_error("error calling memset_s");
            }
#elif defined(SIGMA_USE_EXPLICIT_BZERO)
            explicit_bzero(data, size);
#elif defined(SIGMA_USE_EXPLICIT_MEMSET)
            explicit_memset(data, 0, size);
#else
            volatile sigma_byte *data_ptr = reinterpret_cast<sigma_byte *>(data);
            while (size--)
            {
                *data_ptr++ = sigma_byte{};
            }
#endif
        }
    } // namespace util
} // namespace sigma
