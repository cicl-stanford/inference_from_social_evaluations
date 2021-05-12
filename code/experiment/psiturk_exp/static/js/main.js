//////////////////////////////////////////
//THIS CONTAINS THE JAVASCRIPT FOR A TRIAL
//////////////////////////////////////////

var stage_off = false;
var trial_data = {};
var the_process;
var judged = ["no_ans","no_ans","no_ans","no_ans","no_ans","no_ans","no_ans","no_ans"];//Which queries has the participant responded to?

//For updating the text box next to each slider to the sliders' value
function updateOutput(el, val) {
  el.textContent = val;
}

function Run(current_situation)
{
  //resetting the tracker of what's been answered
  judged = ["no_ans","no_ans","no_ans","no_ans","no_ans","no_ans","no_ans"];
  // adding image
  document.getElementById("real-img").src='../static/images/trials/'+current_situation.base_image

  // making the queries visible
  var unknowns = current_situation.unknowns
  if (unknowns.includes("A-strength")) {
    $( "#A-strength" ).css("visibility", "visible");
    judged[0] = "???";
  }
  if (unknowns.includes("B-strength")) {
    $( "#B-strength" ).css("visibility", "visible");
    judged[1] = "???";
  }
  if (unknowns.includes("C-strength")) {
    $( "#C-strength" ).css("visibility", "visible");
    judged[2] = "???";
  }
  if (unknowns.includes("A-choice")) {
    $( "#A-choice" ).css("visibility", "visible");
    judged[3] = "???";
  }
  if (unknowns.includes("B-choice")) {
    $( "#B-choice" ).css("visibility", "visible");
    judged[4] = "???";
  }
  if (unknowns.includes("C-choice")) {
    $( "#C-choice" ).css("visibility", "visible");
    judged[5] = "???";
  }
  if (unknowns.includes("trees")) {
    $( "#trees" ).css("visibility", "visible");
    judged[6] = "???";
  }
  if (unknowns.includes("fish")) {
    $( "#fish" ).css("visibility", "visible");
    judged[7] = "???";
  }

	console.log('running trial: ', current_situation, '\n');
}

function checkFinished(changed) {
  if (changed === 0) {
    judged[0] = $("#A-strength").val();
  }
  if (changed === 1) {
    judged[1] = $("#B-strength").val();
  }
  if (changed === 2) {
    judged[2] = $("#C-strength").val();
  }
  if (changed === 3) {
    judged[3] = $("#A-choice").val();
  }
  if (changed === 4) {
    judged[4] = $("#B-choice").val();
  }
  if (changed === 5) {
    judged[5] = $("#C-choice").val();
  }
  if (changed === 6) {
    judged[6] = $("#trees").val();
  }
  if (changed === 7) {
    judged[7] = $("#fish").val();
  }
  EnableContinue();
}

function Stop()
{
	EnableContinue();
	
	console.log('stopping!', trial_data);

	clearInterval(the_process);
	clearInterval(countdown_timer);
	// SaveData(trial_data); //Now handled in task.js
}

//Unlock continue when the user has done what they need to
//ZD notes: this needs to be if all visible questions have been answered
function EnableContinue()
{
	if (judged.every(e => e!=="???"))
	{
		$('#next_trial').attr('disabled', false);
	} 
  else 
  {
    $('#next_trial').attr('disabled', true);
  }
}

