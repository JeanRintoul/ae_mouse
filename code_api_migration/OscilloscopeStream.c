/**
 * OscilloscopeStream.c
 *
 * This example performs a stream mode measurement and writes the data to OscilloscopeStream.csv.
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

  // Try to open an oscilloscope with with stream measurement support:
  tiepie_hw_handle scp = TIEPIE_HW_HANDLE_INVALID;

  for(uint32_t index = 0; index < tiepie_hw_devicelist_get_count(); index++)
  {
    const tiepie_hw_handle item = tiepie_hw_devicelist_get_item_by_index(index);

    if(tiepie_hw_devicelistitem_can_open(item, TIEPIE_HW_DEVICETYPE_OSCILLOSCOPE))
    {
      scp = tiepie_hw_devicelistitem_open_oscilloscope(item);
      CHECK_LAST_STATUS();

      // Check for valid handle and stream measurement support:
      if(scp != TIEPIE_HW_HANDLE_INVALID && (tiepie_hw_oscilloscope_get_measure_modes(scp) & TIEPIE_HW_MM_STREAM))
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

    // Set measure mode:
    tiepie_hw_oscilloscope_set_measure_mode(scp, TIEPIE_HW_MM_STREAM);

    // Set sample frequency:
    tiepie_hw_oscilloscope_set_sample_rate(scp, 1e3); // 1 kHz

    // Set record length:
    const uint64_t recordLength = tiepie_hw_oscilloscope_set_record_length(scp, 1000); // 1 kS

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

    // Print oscilloscope info:
    printDeviceInfo(scp);

    // Create data buffers:
    float** channelData = malloc(sizeof(float*) * channelCount);
    for(uint16_t ch = 0; ch < channelCount; ch++)
    {
      channelData[ch] = malloc(sizeof(float) * recordLength);
    }

    // Open file with write/update permissions:
    const char* filename = "OscilloscopeStream.csv";
    FILE* csv = fopen(filename, "w");
    if(csv)
    {
      // Start measurement:
      tiepie_hw_oscilloscope_start(scp);

      // Write csv header:
      fprintf(csv, "Sample");
      for(uint16_t ch = 0; ch < channelCount; ch++)
      {
        fprintf(csv, ";Ch%" PRIu16, (ch + 1));
      }
      fprintf(csv, "\n");

      uint64_t currentSample = 0;

      for(uint8_t chunk = 0; chunk < 10; chunk++) // Measure 10 chunks
      {
        // Print a message, to inform the user that we still do something:
        printf("Data chunk %" PRIu8 "\n", chunk + 1);

        // Wait for measurement to complete:
        while(!(tiepie_hw_oscilloscope_is_data_ready(scp) || tiepie_hw_oscilloscope_is_data_overflow(scp) || tiepie_hw_object_is_removed(scp)))
        {
          sleepMilliSeconds(10); // 10 ms delay, to save CPU time.
        }

        // Print error on device remove:
        if(tiepie_hw_object_is_removed(scp))
        {
          fprintf(stderr, "Device gone!\n");
          status = EXIT_FAILURE;
          break;
        }

        // Print error on data overflow:
        if(tiepie_hw_oscilloscope_is_data_overflow(scp))
        {
          fprintf(stderr, "Data overflow!\n");
          status = EXIT_FAILURE;
          break;
        }

        // Get data:
        uint64_t samplesRead = tiepie_hw_oscilloscope_get_data(scp, channelData, channelCount, 0, recordLength);

        // Write the data to csv:
        for(uint64_t i = 0; i < recordLength; i++)
        {
          fprintf(csv, "%" PRIu64, currentSample + i);
          for(uint16_t ch = 0; ch < channelCount; ch++)
          {
            fprintf(csv, ";%f", channelData[ch][i]);
          }
          fprintf(csv, "\n");
        }

        currentSample += samplesRead;
      }

      printf("Data written to: %s\n", filename);

      // Close file:
      fclose(csv);

      // Stop measurement:
      tiepie_hw_oscilloscope_stop(scp);
    }
    else
    {
      fprintf(stderr, "Couldn't open file: %s\n", filename);
      status = EXIT_FAILURE;
    }

    // Delete data buffers:
    for(uint16_t ch = 0; ch < channelCount; ch++)
      free(channelData[ch]);
    free(channelData);

    // Close oscilloscope:
    tiepie_hw_object_close(scp);
    CHECK_LAST_STATUS();
  }
  else
  {
    fprintf(stderr, "No oscilloscope available with stream measurement support!\n");
    status = EXIT_FAILURE;
  }

  // Exit library:
  tiepie_hw_fini();

  return status;
}
