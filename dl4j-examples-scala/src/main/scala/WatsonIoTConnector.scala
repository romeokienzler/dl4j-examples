import java.util.Properties
import com.google.gson.JsonObject
import com.ibm.iotf.client.app.ApplicationClient
import com.ibm.iotf.client.app.ApplicationStatus
import com.ibm.iotf.client.app.Command
import com.ibm.iotf.client.app.DeviceStatus
import com.ibm.iotf.client.app.Event
import com.ibm.iotf.client.app.EventCallback
import com.ibm.iotf.client.app.StatusCallback

class WatsonIoTConnector(eventbk: EventCallback) {
  val props = new Properties()
  props.load(getClass.getResourceAsStream("/ibm_watson_iot_mqtt.properties"))
  val myClient = new ApplicationClient(props)
  myClient.connect

  val deviceType = "0.16.2"
  val deviceId = "lorenz"

  myClient.setEventCallback(eventbk)

  // Add status callback
  val statusbk = new StatusCallback() {

    override def processDeviceStatus(arg0: DeviceStatus) = {
      System.out.println(arg0)
    }

    override def processApplicationStatus(arg0: ApplicationStatus) {
      System.out.println(arg0)
    }
  }
  myClient.setStatusCallback(statusbk)

  // Subscribe to device events and device connectivity status
  // myClient.subscribeToDeviceStatus();
  myClient.subscribeToDeviceEvents(deviceType, deviceId)

  var count = 0
  // wait for sometime before checking
  while (true) {
    try {
      Thread.sleep(10000)
      println("Mainthread blocking...");
    } catch {
      case e: InterruptedException => {
        e.printStackTrace
      }
    }
  }
}

