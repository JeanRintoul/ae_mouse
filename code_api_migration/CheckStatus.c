/**
 * CheckStatus.c
 *
 * This file is part of the LibTiePie programming examples.
 *
 * Find more information on https://www.tiepie.com/libtiepie-hw-sdk .
 */

#include "CheckStatus.h"
#include <libtiepie-hw.h>
#include <stdio.h>

void checkLastStatus(const char* file, unsigned int line)
{
  tiepie_hw_status status = tiepie_hw_get_last_status();

  if(status < TIEPIE_HW_STATUS_SUCCESS)
    fprintf(stderr, "%s:%u Error: %s\n", file, line, tiepie_hw_get_last_status_str());
  else if(status > TIEPIE_HW_STATUS_SUCCESS)
    fprintf(stderr, "%s:%u Warning: %s\n", file, line, tiepie_hw_get_last_status_str());
}
