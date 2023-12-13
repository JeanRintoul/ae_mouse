/**
 * OscilloscopeConnectionTest.c
 *
 * This example performs a connection test.
 *
 * Find more information on https://www.tiepie.com/libtiepie-hw-sdk .
 */

#include "CheckStatus.h"
#include "PrintInfo.h"
#include "Utils.h"
#include <inttypes.h>
#include <libtiepie-hw.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
  int status = EXIT_SUCCESS;

  // Initialize library:
  tiepie_hw_init();

  // Print library information:
  printLibraryInfo();

  // Enable network search:
  tiepie_hw_network_set_auto_detect_enabled(TIEPIE_HW_BOOL_TRUE);
  CHECK_LAST_STATUS();

  // Update device list:
  tiepie_hw_devicelist_update();
  CHECK_LAST_STATUS();

  // Try to open an oscilloscope with SureConnect support:
  tiepie_hw_handle scp = TIEPIE_HW_HANDLE_INVALID;

  for(uint32_t index = 0; index < tiepie_hw_devicelist_get_count(); index++)
  {
    const tiepie_hw_handle item = tiepie_hw_devicelist_get_item_by_index(index);

    if(tiepie_hw_devicelistitem_can_open(item, TIEPIE_HW_DEVICETYPE_OSCILLOSCOPE))
    {
      scp = tiepie_hw_devicelistitem_open_oscilloscope(item);
      CHECK_LAST_STATUS();

      // Check for valid handle and SureConnect support:
      if(scp != TIEPIE_HW_HANDLE_INVALID && tiepie_hw_oscilloscope_has_sureconnect(scp))
      {
        tiepie_hw_object_close(item);
        break;
      }
      else
      {
        scp = TIEPIE_HW_HANDLE_INVALID;
      }
    }

    tiepie_hw_object_close(item);
  }

  if(scp != TIEPIE_HW_HANDLE_INVALID)
  {
    // Get the number of channels:
    const uint16_t channelCount = tiepie_hw_oscilloscope_get_channel_count(scp);
    CHECK_LAST_STATUS();

    // Enable all channels that support connection testing:
    for(uint16_t ch = 0; ch < channelCount; ch++)
    {
      tiepie_hw_bool b = tiepie_hw_oscilloscope_channel_has_sureconnect(scp, ch);
      CHECK_LAST_STATUS();
      tiepie_hw_oscilloscope_channel_set_enabled(scp, ch, b);
      CHECK_LAST_STATUS();
    }

    // Start connection test on current active channels:
    tiepie_hw_oscilloscope_start_sureconnect(scp);
    CHECK_LAST_STATUS();

    // Wait for connection test to complete:
    while(!tiepie_hw_oscilloscope_is_sureconnect_completed(scp) && !tiepie_hw_object_is_removed(scp))
    {
      sleepMilliSeconds(10); // 10 ms delay, to save CPU time.
    }

    // Create data buffer:
    tiepie_hw_tristate* data = malloc(sizeof(tiepie_hw_tristate) * channelCount);

    // Get data:
    tiepie_hw_oscilloscope_get_sureconnect_data(scp, data, channelCount);
    CHECK_LAST_STATUS();

    // Print results:
    printf("Connection test result:\n");
    for(uint16_t ch = 0; ch < channelCount; ch++)
    {
      printf("Ch%" PRIu16 " = ", ch + 1);

      switch(data[ch])
      {
        case TIEPIE_HW_TRISTATE_UNDEFINED:
          printf("undefined\n");
          break;

        case TIEPIE_HW_TRISTATE_FALSE:
          printf("false\n");
          break;

        case TIEPIE_HW_TRISTATE_TRUE:
          printf("true\n");
          break;

        default:
          printf("unknown state\n");
      }
    }

    // Free data buffer:
    free(data);

    // Close oscilloscope:
    tiepie_hw_object_close(scp);
    CHECK_LAST_STATUS();
  }
  else
  {
    fprintf(stderr, "No oscilloscope available with connection test support!\n");
    status = EXIT_FAILURE;
  }

  // Exit library:
  tiepie_hw_fini();

  return status;
}
