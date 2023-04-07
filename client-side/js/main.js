const gradioBtn = document.querySelector("button");

gradioBtn.addEventListener("click", getGradio);

function getGradio() {
  fetch("https://e47c25ca7e6658ae1a.gradio.live")
    .then((response) => response.json())
    .then((data) => console.log(data))
    .catch((err) => console.log(err));
}
