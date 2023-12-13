/**
 * OscilloscopeGeneratorTrigger.c
 *
 * This example sets up the generator to generate a 1 kHz triangle waveform, 4 Vpp.
 * It also sets up the oscilloscope to perform a block mode measurement, triggered on "Generator new period".
 * A measurement is performed and the data is written to OscilloscopeGeneratorTrigger.csv.
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

  // Try to open an oscilloscope with block measurement support and a generator in the same device:
  tiepie_hw_handle scp = TIEPIE_HW_HANDLE_INVALID;
  tiepie_hw_handle gen = TIEPIE_HW_HANDLE_INVALID;

  for(uint32_t index = 0; index < tiepie_hw_devicelist_get_count(); index++)
  {
    const tiepie_hw_handle item = tiepie_hw_devicelist_get_item_by_index(index);

    if(tiepie_hw_devicelistitem_can_open(item, TIEPIE_HW_DEVICETYPE_OSCILLOSCOPE) && tiepie_hw_devicelistitem_can_open(item, TIEPIE_HW_DEVICETYPE_GENERATOR))
    {
      scp = tiepie_hw_devicelistitem_open_oscilloscope(item);
      CHECK_LAST_STATUS();

      // Check for valid handle and block measurement support:
      if(scp != TIEPIE_HW_HANDLE_INVALID && (tiepie_hw_oscilloscope_get_measure_modes(scp) & TIEPIE_HW_MM_BLOCK))
      {
        gen = tiepie_hw_devicelistitem_open_generator(item);
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

  if(scp != TIEPIE_HW_HANDLE_INVALID && gen != TIEPIE_HW_HANDLE_INVALID)
  {
    // Oscilloscope settings:

    const uint16_t channelCount = tiepie_hw_oscilloscope_get_channel_count(scp);
    CHECK_LAST_STATUS();

    // Set measure mode:
    tiepie_hw_oscilloscope_set_measure_mode(scp, TIEPIE_HW_MM_BLOCK);

    // Set sample frequency:
    tiepie_hw_oscilloscope_set_sample_rate(scp, 1e6); // 1 MHz

    // Set record length:
    uint64_t recordLength = tiepie_hw_oscilloscope_set_record_length(scp, 10000); // 10 kS
    CHECK_LAST_STATUS();

    // Set pre sample ratio:
    tiepie_hw_oscilloscope_set_pre_sample_ratio(scp, 0); // 0 %

    // For all channels:
    for(uint16_t ch = 0; ch < channelCount; ch++)
    {
      // Enable channel to measure it:
      tiepie_hw_oscilloscope_channel_set_enabled(scp, ch, TIEPIE_HW_BOOL_TRUE);
      CHECK_LAST_STATUS();

      // Set range:
      tiepie_hw_oscilloscope_channel_set_range(scp, ch, 8); // 8 V
      CHECK_LAST_STATUS();

      // Set coupling:
      tiepie_hw_oscilloscope_channel_set_coupling(scp, ch, TIEPIE_HW_CK_DCV); // DC Volt
      CHECK_LAST_STATUS();
    }

    // Set trigger timeout:
    tiepie_hw_oscilloscope_trigger_set_timeout(scp, 1); // 1 s
    CHECK_LAST_STATUS();

    // Disable all channel trigger sources:
    for(uint16_t ch = 0; ch < channelCount; ch++)
    {
      tiepie_hw_oscilloscope_channel_trigger_set_enabled(scp, ch, TIEPIE_HW_BOOL_FALSE);
      CHECK_LAST_STATUS();
    }

    // Locate trigger input:
    const uint16_t index = tiepie_hw_device_trigger_get_input_index_by_id(scp, TIEPIE_HW_TIID_GENERATOR_NEW_PERIOD); // or TIID_GENERATOR_START || TIID_GENERATOR_STOP
    CHECK_LAST_STATUS();

    if(index != TIEPIE_HW_TRIGGERIO_INDEX_INVALID)
    {
      // Enable trigger input:
      tiepie_hw_device_trigger_input_set_enabled(scp, index, TIEPIE_HW_BOOL_TRUE);
      CHECK_LAST_STATUS();
    }

    // Generator settings:

    // Set signal type:
    tiepie_hw_generator_set_signal_type(gen, TIEPIE_HW_ST_TRIANGLE);
    CHECK_LAST_STATUS();

    // Set frequency:
    tiepie_hw_generator_set_frequency(gen, 1e3); // 1 kHz
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

    // Print oscilloscope info:
    printDeviceInfo(scp);

    // Print generator info:
    printDeviceInfo(gen);

    // Start measurement:
    tiepie_hw_oscilloscope_start(scp);
    CHECK_LAST_STATUS();

    // Start signal generation:
    tiepie_hw_generator_start(gen);
    CHECK_LAST_STATUS();

    // Wait for measurement to complete:
    while(!tiepie_hw_oscilloscope_is_data_ready(scp) && !tiepie_hw_object_is_removed(scp))
    {
      sleepMilliSeconds(10); // 10 ms delay, to save CPU time.
    }

    if(tiepie_hw_object_is_removed(scp))
    {
      fprintf(stderr, "Device gone!");
      status = EXIT_FAILURE;
    }

    // Stop generator:
    tiepie_hw_generator_stop(gen);
    CHECK_LAST_STATUS();

    // Disable output:
    tiepie_hw_generator_set_output_enable(gen, TIEPIE_HW_BOOL_FALSE);
    CHECK_LAST_STATUS();

    if(tiepie_hw_oscilloscope_is_data_ready(scp))
    {
      // Create data buffers:
      float** channelData = malloc(sizeof(float*) * channelCount);
      for(uint16_t ch = 0; ch < channelCount; ch++)
      {
        channelData[ch] = malloc(sizeof(float) * recordLength);
      }

      // Get the data from the scope:
      recordLength = tiepie_hw_oscilloscope_get_data(scp, channelData, channelCount, 0, recordLength);
      CHECK_LAST_STATUS();

      // Open file with write/update permissions:
      const char* filename = "OscilloscopeGeneratorTrigger.csv";
      FILE* csv = fopen(filename, "w");
      if(csv)
      {
        // Write csv header:
        fprintf(csv, "Sample");
        for(uint16_t ch = 0; ch < channelCount; ch++)
        {
          fprintf(csv, ";Ch%" PRIu16, ch + 1);
        }
        fprintf(csv, "\n");

        // Write the data to csv:
        for(uint64_t i = 0; i < recordLength; i++)
        {
          fprintf(csv, "%" PRIu64, i);
          for(uint16_t ch = 0; ch < channelCount; ch++)
          {
            fprintf(csv, ";%f", channelData[ch][i]);
          }
          fprintf(csv, "\n");
        }

        printf("Data written to: %s\n", filename);

        // Close file:
        fclose(csv);
      }
      else
      {
        fprintf(stderr, "Couldn't open file: %s\n", filename);
        status = EXIT_FAILURE;
      }

      // Free data buffers:
      for(uint16_t ch = 0; ch < channelCount; ch++)
      {
        free(channelData[ch]);
      }
      free(channelData);
    }

    // Close oscilloscope:
    tiepie_hw_object_close(scp);
    CHECK_LAST_STATUS();

    // Close generator:
    tiepie_hw_object_close(gen);
    CHECK_LAST_STATUS();
  }
  else
  {
    fprintf(stderr, "No oscilloscope available with block measurement support or generator available in the same unit!\n");
    status = EXIT_FAILURE;
  }

  // Exit library:
  tiepie_hw_fini();

  return status;
}
