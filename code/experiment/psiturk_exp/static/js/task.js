/*
 * Requires:
 *     psiturk.js
 *     utils.js
 */

// Initalize psiturk object
var psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);

//var mycondition = condition;  // these two variables are passed by the psiturk server process
condition = Math.floor(Math.random() * 4);  //returns a random number from 0 to 3
var mycondition = condition;
var mycounterbalance = counterbalance;  // they tell you which condition you have been assigned to

// All pages to be loaded
var pages = [
	"instructions/instruct-1.html",
	"instructions/instruct-2.html",
	"instructions/instruct-3.html",
	"instructions/instruct-4.html",
	"instructions/instruct-5.html",
	"instructions/instruct-6.html",
	"instructions/instruct-7.html",
	"instructions/instruct-8.html",
	"instructions/instruct-9.html",
	"instructions/instruct-10.html",
	"instructions/instruct-11.html",
	"instructions/instruct-12.html",
	"instructions/instruct-13.html",
	"instructions/instruct-14.html",
	"instructions/instruct-15.html",
	"instructions/comprehension-01.html",
	"instructions/comprehension-02.html",
	"instructions/comprehension-03.html",
	"instructions/comprehension-04.html",
	"instructions/comprehension-05.html",
	"instructions/comprehension-06.html",
	"instructions/comprehension-07.html",
	"instructions/comprehension-08.html",
	"instructions/comprehension-09.html",
	"instructions/comprehension-10.html",
	"instructions/comprehension-11.html",
	"instructions/comprehension-12.html",
	"instructions/instruct-ready.html",
	"comprehension-check.html",
	"interim-instruct.html",
	"stage.html",
	"postquestionnaire.html"
];

psiTurk.preloadPages(pages);

var instructionPages = [ // add as a list as many pages as you like
	/*
	*/
	"instructions/instruct-1.html",
	"instructions/instruct-2.html",
	"instructions/instruct-3.html",
	"instructions/instruct-4.html",
	"instructions/instruct-5.html",
	"instructions/instruct-6.html",
	"instructions/instruct-7.html",
	"instructions/instruct-8.html",
	"instructions/instruct-9.html",
	"instructions/instruct-10.html",
	"instructions/instruct-11.html",
	"instructions/instruct-12.html",
	"instructions/instruct-13.html",
	"instructions/instruct-14.html",
	"instructions/instruct-15.html",
	"instructions/comprehension-01.html",
	"instructions/comprehension-02.html",
	"instructions/comprehension-03.html",
	"instructions/comprehension-04.html",
	"instructions/comprehension-05.html",
	"instructions/comprehension-06.html",
	"instructions/comprehension-07.html",
	"instructions/comprehension-08.html",
	"instructions/comprehension-09.html",
	"instructions/comprehension-10.html",
	"instructions/comprehension-11.html",
	"instructions/comprehension-12.html"
];

/********************
* The Experiment       *
********************/
var TheExperiment = function() {

	//Keep a note of the current trial
	cur_trial = 0;

	//Shuffle the array of situations to randomise the trial order
	situations = _.shuffle(situations);
	//inserting attention checks
	situations.splice(11, 0, {"id": ["att1"], "base_image": "attention1.png", "unknowns": ["fish"]});
	situations.splice(23, 0, {"id": ["att2"], "base_image": "attention2.png", "unknowns": ["fish"]});

	//Set up the trial structure
	var n_trials = situations.length;

	console.log('condition', condition, 'counter', counterbalance);
	
	next = function() {
		n_trials=situations.length;

		if (cur_trial === situations.length) {
			//Saving condition
			psiTurk.recordTrialData({"real_condition": condition})

			finish();
		} else {
			//Print the current trial info to the console
			console.log('cur_trial:', cur_trial, 'n_trials', n_trials);
			next_trial();
		}

	};

	//Run a trial
	next_trial = function () {
		//Select the current network (this is a list with two elements)
		current_situation = situations[cur_trial];

		//Show the trial slide	
		psiTurk.showPage('stage.html');
		Run(current_situation);

		//Run the OU process code using the current network settings
		//Run(current_network.betas);

		cur_trial = cur_trial + 1;

		//Add progress
		var elem = document.getElementById("myprogressbar");
		elem.style.width = 100*(cur_trial/n_trials) + '%';
		elem.innerHTML =  'Situation ' + cur_trial + ' of ' + n_trials;
		//d3.select("#progress-bar").style.width = (cur_trial/n_trials) + '%';
		d3.select("#stage_heading").html('Situation ' + cur_trial + ' of ' + n_trials + '.</p>');
	}

	var finish = function() {
	    currentview = new Questionnaire();
	};


	// Load the stage.html snippet into the body of the page
	psiTurk.showPage('stage.html');

	// Start the test
	next();
};

var endTrial = function() {
	//They will probably change their judgment after the 45 seconds is up so this reads the final choice
	//Save data NOTE: needs to be only when continue is pressed
	var A_strength = document.getElementById("A-strength").value;
	var B_strength = document.getElementById("B-strength").value;
	var C_strength = document.getElementById("C-strength").value;
	var A_choice = document.getElementById("A-choice").value;
	var B_choice = document.getElementById("B-choice").value;
	var C_choice = document.getElementById("C-choice").value;
	var trees = document.getElementById("trees").value;
	var fish = document.getElementById("fish").value;

	console.log(A_strength, B_strength, C_strength, 
		A_choice, B_choice, C_choice, trees, fish)

	psiTurk.recordTrialData({"phase": "TESTTRIAL",
							"condition": condition,
	                        "trial_ix":current_situation.id,
	                        "trial_number":cur_trial,
	                        "situation":current_situation,
	                        "A_strength": A_strength,
							"B_strength": B_strength,
							"C_strength": C_strength,
							"A_choice": A_choice,
							"B_choice": B_choice,
							"C_choice": C_choice,
							"trees": trees,
							"fish": fish});
	
	console.log('end trial function fired');

	next();
}

// var nextBlock = function () {
// 	psiTurk.hidePage('interim-instruct.html');
// 	psiTurk.showPage('stage.html');
// }

/*****************
 *  COMPREHENSION CHECK*
 *****************/
var Comprehension = function(){
	psiTurk.showPage('comprehension-check.html')

    //disable button initially
    //$('#done_comprehension').prop('disabled', false); ////////changed from true

    $('#done_comprehension').click(function () { 
		currentview = new TheExperiment();
	});
};

/*****************
 *  COMPREHENSION FAIL SCREEN*
 *****************/

var ComprehensionCheckFail = function(){
// Show the slide
$(".slide").hide();
$("#comprehension_check_fail").fadeIn($c.fade);

$('#comprehension_fail').click(function () {           
    $('#comprehension_fail').unbind();
    currentview = new Instructions();
   });
}


/****************
* Post test questionnaire *
****************/

var Questionnaire = function() {

	var error_message = "<h1>Oops!</h1><p>Something went wrong submitting your HIT. " + 
	"This might happen if you lose your internet connection. Press the button to resubmit.</p><button id='resubmit'>Resubmit</button>";

	posttest_button_disabler = function () {
		if($("#feedback").val() === '' || $("#age").val() === '' || $("#sex").val() === 'noresp' || $("#engagement").val() === '--' || $("#difficulty").val() === '--') {
			$('#done_posttest').prop('disabled',true);
		} else{
			$('#done_posttest').prop('disabled',false);
		}
	}

	record_responses = function() {
		if($("#feedback").val() === '' || $("#age").val() === '' || $("#sex").val() === 'noresp' || $("#engagement").val() === '--' || $("#difficulty").val() === '--') {
			alert('You must fill all forms');
		} else{
			psiTurk.recordTrialData({'phase':'POSTTEST', 'status':'submit'});

			$('textarea').each( function(i, val) {
				psiTurk.recordUnstructuredData(this.id, this.value);
			});
			$('select').each( function(i, val) {
				psiTurk.recordUnstructuredData(this.id, this.value);		
			});
			$('input').each( function(i, val) {
				psiTurk.recordUnstructuredData(this.id, this.value);		
			});

			//saving the data
			psiTurk.saveData({
			success: function(){
                psiTurk.completeHIT(); // when finished saving compute bonus, the quit
			}, 
			error: prompt_resubmit});
		}
	};

	prompt_resubmit = function() {
		document.body.innerHTML = error_message;
		$("#resubmit").click(resubmit);
	};

	resubmit = function() {
		document.body.innerHTML = "<h1>Trying to resubmit...</h1>";
		reprompt = setTimeout(prompt_resubmit, 10000);
		
		psiTurk.saveData({
			success: function() {
				clearInterval(reprompt); 
				psiTurk.computeBonus('compute_bonus', function(){finish()}); 
			}, 
			error: prompt_resubmit
		});
	};

	// Load the questionnaire snippet 
	psiTurk.showPage('postquestionnaire.html');
	psiTurk.recordTrialData({'phase':'POSTTEST', 'status':'begin'});
	
	$(".posttestQ").change(function () {
		posttest_button_disabler();
	})

	$("#done_posttest").click(function () {
		record_responses();
	});

	
};

// Task object to keep track of the current phase
var currentview;

/*******************
 * Run Task
 ******************/
 $(window).load( function(){
 	psiTurk.doInstructions(
    	instructionPages, // a list of pages you want to display in sequence
    	function() {
	    	// Load the data from the json then start the exp
	    	$.getJSON('/static/json/stim.json', function (data) {
	    		console.log(data);
	    		situations = data.situations;
	      		// Load the experiment configuration from the server
	      		currentview = new TheExperiment();
	      	});
     } // what you want to do when you are done with instructions
     );
 });
