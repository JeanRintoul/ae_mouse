/**
 * OscilloscopeCombineHS3HS4.c
 *
 * This example demonstrates how to create and open a combined instrument of all found Handyscope HS3, Handyscope HS4 and/or Handyscope HS4 DIFF's.
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

  // Update device list:
  tiepie_hw_devicelist_update();
  CHECK_LAST_STATUS();

  // Allocate memory for storing device handles:
  uint32_t deviceCount = tiepie_hw_devicelist_get_count();
  CHECK_LAST_STATUS();
  tiepie_hw_handle* deviceHandles = malloc(sizeof(tiepie_hw_handle) * deviceCount);
  uint32_t deviceHandleCount = 0;
  tiepie_hw_handle scp = TIEPIE_HW_HANDLE_INVALID;

  // Try to open all HS3/HS4(D) oscilloscopes:
  for(uint32_t index = 0; index < deviceCount; index++)
  {
    const tiepie_hw_handle item = tiepie_hw_devicelist_get_item_by_index(index);

    uint32_t productId = tiepie_hw_devicelistitem_get_product_id(item);
    CHECK_LAST_STATUS();
    printf("product id: %d\n",productId);
    
    if((productId == TIEPIE_HW_PRODUCTID_HS5 || productId == TIEPIE_HW_PRODUCTID_WS5 || productId == TIEPIE_HW_PRODUCTID_WS6D) &&
       tiepie_hw_devicelistitem_can_open(item, TIEPIE_HW_DEVICETYPE_OSCILLOSCOPE))
    {
      scp = tiepie_hw_devicelistitem_open_oscilloscope(item);
      CHECK_LAST_STATUS();

      if(scp != TIEPIE_HW_HANDLE_INVALID)
      {
        // Get name:
        uint32_t length = tiepie_hw_device_get_name(scp, NULL, 0) + 1; // Add one for the terminating zero
        CHECK_LAST_STATUS();
        char* name = malloc(sizeof(char) * length);
        length = tiepie_hw_device_get_name(scp, name, length);
        CHECK_LAST_STATUS();

        printf("Found: %s, s/n: %u\n", name, tiepie_hw_device_get_serial_number(scp));
        CHECK_LAST_STATUS();

        free(name);

        deviceHandles[deviceHandleCount] = scp;
        deviceHandleCount++;
      }
    }

    tiepie_hw_object_close(item);
  }

  if(deviceHandleCount > 1)
  {
    // Create and open combined instrument:
    tiepie_hw_handle scp = tiepie_hw_devicelist_create_and_open_combined_device(deviceHandles, deviceHandleCount);
    CHECK_LAST_STATUS();

    // Release HS3/HS4(D) handles, not required anymore:
    for(uint32_t i = 0; i < deviceHandleCount; i++)
    {
      tiepie_hw_object_close(deviceHandles[i]);
      CHECK_LAST_STATUS();
    }
    free(deviceHandles);
    deviceHandles = NULL;

    // Print combined oscilloscope info:
    printDeviceInfo(scp);

    // Get serial number, required for removing:
    uint32_t serialNumber = tiepie_hw_device_get_serial_number(scp);
    CHECK_LAST_STATUS();

    // Close combined oscilloscope:
    tiepie_hw_object_close(scp);
    CHECK_LAST_STATUS();

    // Remove combined oscilloscope from the device list:
    tiepie_hw_devicelist_remove_device(serialNumber, TIEPIE_HW_BOOL_FALSE);
    CHECK_LAST_STATUS();
  }
  else
  {
    fprintf(stderr, "Not enough HS5/WS6(D)\'s found, at least two required!\n");
    status = EXIT_FAILURE;
  }

  // Release HS3/HS4(D) handles:
  if(deviceHandles)
  {
    for(uint32_t i = 0; i < deviceHandleCount; i++)
    {
      tiepie_hw_object_close(deviceHandles[i]);
    }
    free(deviceHandles);
  }

  // Exit library:
  tiepie_hw_fini();

  return status;
}
