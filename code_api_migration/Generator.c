/**
 * Generator.c
 *
 * This example generates a 100 kHz triangle waveform, 4 Vpp.
 *
 * Find more information on https://www.tiepie.com/libtiepie-hw-sdk .
 */

#include "CheckStatus.h"
#include "PrintInfo.h"
#include "Utils.h"
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

  // Try to open a generator:
  tiepie_hw_handle gen = TIEPIE_HW_HANDLE_INVALID;

  for(uint32_t index = 0; index < tiepie_hw_devicelist_get_count(); index++)
  {
    const tiepie_hw_handle item = tiepie_hw_devicelist_get_item_by_index(index);

    if(tiepie_hw_devicelistitem_can_open(item, TIEPIE_HW_DEVICETYPE_GENERATOR))
    {
      gen = tiepie_hw_devicelistitem_open_generator(item);
      CHECK_LAST_STATUS();

      // Check for valid handle:
      if(gen != TIEPIE_HW_HANDLE_INVALID)
      {
        tiepie_hw_object_close(item);
        break;
      }
    }

    tiepie_hw_object_close(item);
  }

  if(gen != TIEPIE_HW_HANDLE_INVALID)
  {
    // Set signal type:
    tiepie_hw_generator_set_signal_type(gen, TIEPIE_HW_ST_TRIANGLE);
    CHECK_LAST_STATUS();

    // Set frequency:
    tiepie_hw_generator_set_frequency(gen, 100e3); // 100 kHz
    CHECK_LAST_STATUS();

    // Set amplitude:
    tiepie_hw_generator_set_amplitude(gen, 2); // 2 V
    CHECK_LAST_STATUS();

    // Set offset:
    tiepie_hw_generator_set_offset(gen, 0); // 0 V
    CHECK_LAST_STATUS();

    // Enable output:
    tiepie_hw_generator_set_output_enable(gen, TIEPIE_HW_BOOL_TRUE);
    CHECK_LAST_STATUS();

    // Print generator info:
    printDeviceInfo(gen);

    // Start signal generation:
    tiepie_hw_generator_start(gen);
    CHECK_LAST_STATUS();

    // Wait for keystroke:
    printf("Press any key to stop signal generation...\n");
    waitForKeyStroke();

    // Stop generator:
    tiepie_hw_generator_stop(gen);
    CHECK_LAST_STATUS();

    // Disable output:
    tiepie_hw_generator_set_output_enable(gen, TIEPIE_HW_BOOL_FALSE);
    CHECK_LAST_STATUS();

    // Close generator:
    tiepie_hw_object_close(gen);
    CHECK_LAST_STATUS();
  }
  else
  {
    fprintf(stderr, "No generator available!\n");
    status = EXIT_FAILURE;
  }

  // Exit library:
  tiepie_hw_fini();

  return status;
}