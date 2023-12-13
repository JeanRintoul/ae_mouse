/**
 * GeneratorTriggeredBurst.c
 *
 * This example generates a 100 kHz square waveform, 25% duty cycle, 0..5 V, 20 periods, this waveform is triggered by the external trigger.
 * Connect the external trigger to GND to trigger the burst.
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

  // Try to open a generator with triggered burst support:
  tiepie_hw_handle gen = TIEPIE_HW_HANDLE_INVALID;

  for(uint32_t index = 0; index < tiepie_hw_devicelist_get_count(); index++)
  {
    const tiepie_hw_handle item = tiepie_hw_devicelist_get_item_by_index(index);

    if(tiepie_hw_devicelistitem_can_open(item, TIEPIE_HW_DEVICETYPE_GENERATOR))
    {
      gen = tiepie_hw_devicelistitem_open_generator(item);
      CHECK_LAST_STATUS();

      // Check for valid handle and triggered burst support:
      if(gen != TIEPIE_HW_HANDLE_INVALID && (tiepie_hw_generator_get_modes_native(gen) & TIEPIE_HW_GM_BURST_COUNT) && tiepie_hw_device_trigger_get_input_count(gen) > 0)
      {
        tiepie_hw_object_close(item);
        break;
      }
      else
      {
        gen = TIEPIE_HW_HANDLE_INVALID;
      }
    }

    tiepie_hw_object_close(item);
  }

  if(gen != TIEPIE_HW_HANDLE_INVALID)
  {
    // Set signal type:
    tiepie_hw_generator_set_signal_type(gen, TIEPIE_HW_ST_SQUARE);
    CHECK_LAST_STATUS();

    // Set frequency:
    tiepie_hw_generator_set_frequency(gen, 100e3); // 100 kHz
    CHECK_LAST_STATUS();

    // Set amplitude:
    tiepie_hw_generator_set_amplitude(gen, 2.5); // 2.5 V
    CHECK_LAST_STATUS();

    // Set offset:
    tiepie_hw_generator_set_offset(gen, 2.5); // 2.5 V
    CHECK_LAST_STATUS();

    // Set symmetry (duty cycle):
    tiepie_hw_generator_set_symmetry(gen, 0.25); // 25 %
    CHECK_LAST_STATUS();

    // Set mode:
    tiepie_hw_generator_set_mode(gen, TIEPIE_HW_GM_BURST_COUNT);
    CHECK_LAST_STATUS();

    // Set burst count:
    tiepie_hw_generator_set_burst_count(gen, 20); // 20 periods
    CHECK_LAST_STATUS();

    // Locate trigger input:
    uint16_t index = tiepie_hw_device_trigger_get_input_index_by_id(gen, TIEPIE_HW_TIID_EXT1);
    CHECK_LAST_STATUS();

    if(index == TIEPIE_HW_TRIGGERIO_INDEX_INVALID)
    {
      index = tiepie_hw_device_trigger_get_input_index_by_id(gen, TIEPIE_HW_TIID_EXT2);
      CHECK_LAST_STATUS();
    }

    if(index != TIEPIE_HW_TRIGGERIO_INDEX_INVALID)
    {
      // Enable trigger input:
      tiepie_hw_device_trigger_input_set_enabled(gen, index, TIEPIE_HW_BOOL_TRUE);
      CHECK_LAST_STATUS();

      // Set trigger input kind:
      tiepie_hw_device_trigger_input_set_kind(gen, index, TIEPIE_HW_TK_FALLINGEDGE);
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
      fprintf(stderr, "Unknown trigger input!\n");
      status = EXIT_FAILURE;
    }
  }
  else
  {
    fprintf(stderr, "No generator available with triggererd burst support!\n");
    status = EXIT_FAILURE;
  }

  // Exit library:
  tiepie_hw_fini();

  return status;
}
