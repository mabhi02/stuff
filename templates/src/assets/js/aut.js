// Function to save form data to localStorage for account creation
function createAccount() {
    const name = document.getElementById("exampleInputtext1").value;
    const email = document.getElementById("exampleInputEmail1").value;
    const password = document.getElementById("exampleInputPassword1").value;
  
    // Check if required fields are not empty
    if (!name || !email || !password) {
      alert("Please fill in all the required fields.");
      return false; // Prevent form submission
    }
  
    // Save account data to localStorage
    const userData = {
      name: name,
      email: email,
      password: password
    };
  
    // Generate a unique user ID or use email as a key
    const userId = email.replace(/[^\w]/g, ''); // Remove non-alphanumeric characters
  
    // Save user data with a unique key
    localStorage.setItem(userId, JSON.stringify(userData));
  
    // Clear the form fields for a fresh registration
    document.getElementById("exampleInputtext1").value = "";
    document.getElementById("exampleInputEmail1").value = "";
    document.getElementById("exampleInputPassword1").value = "";
  
    alert("Account created successfully. You can now log in.");
    return false; // Prevent form submission
  }
  
  // Function to load saved user data on page load for create account
  window.onload = function () {
    // Your existing code to load user data on page load
  };
  