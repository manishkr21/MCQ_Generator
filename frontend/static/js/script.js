function shuffle(array) {
    return array.map(value => ({ value, sort: Math.random() }))
    .sort((a, b) => a.sort - b.sort)
    .map(({ value }) => value)
}

function appendData(data) {
    let mainContainer = document.getElementById("myData");
   
    mainContainer.innerHTML = '';
    for (let i = 0; i < data.length; i++) {
        let div_element = document.createElement("div");
        let text = "";
        let choices = data[i]["choices"];
        choices.push(data[i].answer);
        choices = shuffle(choices)
        // console.log(choices);
        let id = 'attrib'
        choices.forEach(function(choice,index) {     text += '<li>'+choice+'</li>'    })  
        div_element.innerHTML = '<div class="mcqs" id=' + id + (i + 1) + '><p> <b>Question ' + (i + 1) + ': ' + data[i].question + "</b><button id='edit_btn' type='button' class='btn btn-outline-primary' onclick = 'to_edit(" + id + (i + 1) + ")'><img src='static/img/edit.svg'>  </button>" + '</p><ol>' + text + '</ol> <p > Correct Answer : <span style="color: green; font-weight:bold">' + data[i].answer + '</span></p></div>';

        mainContainer.appendChild(div_element);
    }
}

let loadingDiv = document.getElementById('loading');
// console.log(loadingDiv, 'loadingDiv');

function showSpinner() {
  loadingDiv.style.visibility = 'visible';
}

function hideSpinner() {
  loadingDiv.style.visibility = 'hidden';
}
hideSpinner();
const load_mcq = () => {
    const user_textarea = document.getElementById("user_text").value;
    showSpinner();
    fetch('http://127.0.0.1:5000/gen', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
                'text': user_textarea
        })
    })
    .then(response => {
        if (response.ok) {
            return response.json()
        } else {
            console.log('response not ok!')
        }
        hideSpinner();
    })
    .then(data => {
        console.log(data)
        appendData(data)
        hideSpinner();
    })
    .catch(err => {
        hideSpinner();
        console.error(err)
    })
};

let flag = 0;

const to_edit=(paragraph_id) => {

    // const paragraph = document.getElementById("res");
    if (flag ==0){
        paragraph_id.contentEditable = true;
        paragraph_id.style.backgroundColor = "#ddd";
        // paragraph_id.style.padding = "5px";

        flag = 1;
    
    }else{
        paragraph_id.contentEditable = false;
        paragraph_id.style.backgroundColor = "#fff";
        // paragraph_id.style.padding = "0px";
        flag =0;
    }

}