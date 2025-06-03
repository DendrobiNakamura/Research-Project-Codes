//https://forum.arduino.cc/t/serial-input-basics-updated/382007/2

// defines pins numbers
#define DirR 2
#define PulseR 3
#define DirL 8
#define PulseL 9
#define InSW1 12 // switch 1 is the furtherest away of the antenna and is analog input 0
#define InSW2 13 // switch 2 is the closest to the antenna and is analog input 1

//Direction == 0 means away from the antenna
//Direction == 1 means towards the antenna

const byte numChars = 32;
char receivedChars[numChars];
char tempChars[numChars];        // temporary array for use when parsing

// variables to hold the parsed data
char MotorID[numChars] = {0};
int Direction = 0;
double Distance = 0.;
double AngleStep = 0;

//1 == open swtich, 0 == closed switch
int switch1;
int switch2;

boolean newData = false;

//============

void setup()
{
  Serial.begin(9600);
  // Sets four pins as Outputs for the motors
  pinMode(DirL, OUTPUT);
  pinMode(PulseL, OUTPUT);
  pinMode(DirR, OUTPUT);
  pinMode(PulseR, OUTPUT);
  // Sets two pins at Inputs for the switches
  pinMode(InSW1, INPUT_PULLUP);
  pinMode(InSW2, INPUT_PULLUP);

  Serial.println("This code expects 1 chars and 2 intergers - the motor ID, the direction and the number of turns / angle of rotation");
  Serial.println("Enter data in this style <L, 1, 10>");
  Serial.println("'L' and 'R' are for the linear and rotation motors respectively.");
  Serial.println("For the linear motor, 1 means towards the antenna and 0 means away from the antenna.");
  Serial.println("For the rotation motor, 1 is clockwise when looking along B field, 0 is anticlockwise.");
}

//============

void loop()
{
  recvWithStartEndMarkers();
  if (newData == true)
  {
    strcpy(tempChars, receivedChars);
    // this temporary copy is necessary to protect the original data
    // because strtok() used in parseData() replaces the commas with \0
    parseData();
    turnInDirByNturns();
    newData = false;
  }
}

//============

void recvWithStartEndMarkers()
{
  static boolean recvInProgress = false;
  static byte ndx = 0;
  char startMarker = '<';
  char endMarker = '>';
  char rc;

  while (Serial.available() > 0 && newData == false) {
    rc = Serial.read();

    if (recvInProgress == true)
    {
      if (rc != endMarker) {
        receivedChars[ndx] = rc;
        ndx++;
        if (ndx >= numChars)
        {
          ndx = numChars - 1;
        }
      }
      else
      {
        receivedChars[ndx] = '\0'; // terminate the string
        recvInProgress = false;
        ndx = 0;
        newData = true;
      }
    }
    else if (rc == startMarker)
    {
      recvInProgress = true;
    }
  }
}

//============

void parseData()  // split the data into its parts
{
  Direction = 0;  //Clean them up at each call
  Distance = 0.;
  AngleStep = 0.;
  char * strtokIndx; // this is used by strtok() as an index

  strtokIndx = strtok(tempChars, ",");     // get the first part - the string
  strcpy(MotorID, strtokIndx);             // copy it to messageFromPC

  strtokIndx = strtok(NULL, ",");         //this continues where the previous call left off
  Direction = atoi(strtokIndx);           // convert this part to an integer

  if ( strcmp(MotorID, "L") == 0 )
  {
    strtokIndx = strtok(NULL, ",");         // this continues where the previous call left off
    Distance = atof(strtokIndx);            // convert this part to a double
  }
  else if ( strcmp(MotorID, "R") == 0 )
  {
    strtokIndx = strtok(NULL, ",");         // this continues where the previous call left off
    AngleStep = atof(strtokIndx);          // convert this part to a double
  }

  //  Serial.println(MotorID);
  //  Serial.println(Direction);
  //  Serial.println(Distance);
  //  Serial.println(MotorID);
  //  Serial.println(Direction);
  //  Serial.println(AngleStep);

}

//============

void turnInDirByNturns()        //Do something with the steppers motors
{

  if ( strcmp(MotorID, "L") == 0 )
  {
    digitalWrite(DirL, Direction); // Enables the linear motor to move in a specified direction
    switch1 = 1;
    switch2 = 1;

    int i = 0;
    for (long x = 0; x < 400.*Distance && switch1 == 1 && switch2 == 1; x++) // Makes 400 pulses for one full cycle rotation == 1cm
    {
      digitalWrite(PulseL, HIGH);
      delayMicroseconds(700);
      digitalWrite(PulseL, LOW);
      delayMicroseconds(700);

      if (i > 5)  //necessary to do 4 steps of the for loop to free the switch when the motor is home
      {
        switch1 = digitalRead(InSW1);
        switch2 = digitalRead(InSW2);
      }
      i += 1;
    }
  }
  else if ( strcmp(MotorID, "R") == 0 )
  {
    digitalWrite(DirR, Direction); // Enables the rotation motor to turn in a specified direction
    //1 is clockwise when looking along B field, 0 is anticlockwise

    for (long x = 0; x < (3200.*AngleStep)/360.; x++) // 3200 steps = 1 full rotation
    {
      digitalWrite(PulseR, HIGH);
      delayMicroseconds(1200);
      digitalWrite(PulseR, LOW);
      delayMicroseconds(1200);
    }

  }
  else
  {
    Serial.println("Something is wrong with the MotorID.");
  }

}
