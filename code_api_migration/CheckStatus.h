/**
 * CheckStatus.h
 *
 * This file is part of the LibTiePie programming examples.
 *
 * Find more information on https://www.tiepie.com/libtiepie-hw-sdk .
 */

#ifndef _CHECKSTATUS_H_
#define _CHECKSTATUS_H_

#define CHECK_LAST_STATUS() checkLastStatus(__FILE__, __LINE__);

void checkLastStatus(const char* file, unsigned int line);

#endif
